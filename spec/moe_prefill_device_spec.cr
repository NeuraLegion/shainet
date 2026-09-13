require "./spec_helper"

# Device-resident MoE PREFILL. Its claim is a TRANSFER BUDGET: the activations are
# uploaded once and read back once per layer, instead of every expert matmul
# bouncing to the host on its own. Parity alone would not catch a regression that
# quietly restored those readbacks, so the counts are asserted directly, as an
# ABSENCE of the per-matmul phases.
private def moe_prefill_ready?
  SHAInet::CUDA.fully_available? && SHAInet::CUDA.swiglu_kernel_available?
end

private def with_moe_device(enabled : Bool?, &)
  prev = SHAInet::MoEFF.device_decode_enabled?
  SHAInet::MoEFF.device_decode_enabled = enabled
  begin
    yield
  ensure
    SHAInet::MoEFF.device_decode_enabled = prev
  end
end

private def with_prof(&)
  prev = SHAInet::Profile.enabled?
  SHAInet::Profile.enabled = true
  SHAInet::Profile.reset
  begin
    yield
  ensure
    SHAInet::Profile.enabled = prev
    SHAInet::Profile.reset
  end
end

private def fill!(m, seed : Float64)
  m.rows.times do |i|
    m.cols.times { |j| m[i, j] = Math.sin(seed + i * 0.23 + j * 0.11) * 0.4 }
  end
  m
end

private def build_moe(d_model : Int32, ff : Int32, experts : Int32, top_k : Int32)
  moe = SHAInet::MoEFF.new(d_model, ff, experts, top_k, true, false)
  fill!(moe.router.as(SHAInet::SimpleMatrix), 0.7)
  moe.experts.each_with_index do |e, i|
    e.gate_proj = fill!(SHAInet::SimpleMatrix.new(d_model, ff), 1.0 + i)
    e.up_proj = fill!(SHAInet::SimpleMatrix.new(d_model, ff), 2.0 + i)
    e.down_proj = fill!(SHAInet::SimpleMatrix.new(ff, d_model), 3.0 + i)
  end
  moe.to_gpu!(true, 4)
  moe
end

describe "device-resident MoE prefill" do
  it "matches the host path over many tokens" do
    pending! "CUDA/kernels not available" unless moe_prefill_ready?

    d_model = 64
    x = fill!(SHAInet::SimpleMatrix.new(12, d_model), 0.3) # 12 tokens: prefill, not decode

    host = with_moe_device(false) { build_moe(d_model, 96, 6, 2).forward(x) }
    dev = with_moe_device(true) { build_moe(d_model, 96, 6, 2).forward(x) }

    host.rows.should eq(dev.rows)
    host.cols.should eq(dev.cols)
    host.rows.times do |i|
      host.cols.times { |j| dev[i, j].should be_close(host[i, j], 1e-3) }
    end
  end

  it "spends two transfers per layer, not two per expert matmul" do
    pending! "CUDA/kernels not available" unless moe_prefill_ready?

    moe = build_moe(64, 96, 6, 2)
    x = fill!(SHAInet::SimpleMatrix.new(12, 64), 0.9)

    with_moe_device(true) do
      with_prof do
        moe.forward(x)
        stats = SHAInet::Profile.stats

        # The budget: ONE upload and ONE readback for the whole 12-token block.
        stats["ffn.prefill_h2d"].not_nil![:count].should eq(1)
        stats["ffn.prefill_d2h"].not_nil![:count].should eq(1)

        # And the per-matmul host round trip must not fire AT ALL. With 12 tokens x
        # 2 experts x 3 matmuls the old path would have logged 72 of each, so this
        # asserted absence is what a regression would trip.
        stats["gemm.out_d2h"]?.should be_nil
        stats["gemm.in_h2d"]?.should be_nil

        # The batching claim. 12 tokens x 2 experts is 24 token-expert assignments,
        # issued as ONE batched GEMM per expert that received tokens (at most 6),
        # not one per assignment. The phases are measured per batch, so this count
        # IS the number of GEMM batches.
        gemms = stats["ffn.batch_gate_up"].not_nil![:count]
        gemms.should be > 0
        gemms.should be <= 6
        gemms.should be < 12 * 2
        stats["ffn.batch_down"].not_nil![:count].should eq(gemms)

        # And no token took the per-token single-row expert path, which is the
        # thing being replaced. Asserted as an absence.
        stats["ffn.dev_gate_up"]?.should be_nil
      end
    end
  end

  it "still uses the host path for a single token" do
    pending! "CUDA/kernels not available" unless moe_prefill_ready?

    # Decode has its own tuned single-row path; the prefill wrapper must decline so
    # it does not take over and regress it.
    moe = build_moe(64, 96, 6, 2)
    x = fill!(SHAInet::SimpleMatrix.new(1, 64), 0.5)

    with_moe_device(true) do
      with_prof do
        moe.forward(x)
        SHAInet::Profile.stats["ffn.prefill_h2d"]?.should be_nil
      end
    end
  end

  it "falls back to the host path when the device path is disabled" do
    pending! "CUDA/kernels not available" unless moe_prefill_ready?

    moe = build_moe(64, 96, 6, 2)
    x = fill!(SHAInet::SimpleMatrix.new(12, 64), 0.4)

    with_moe_device(false) do
      with_prof do
        moe.forward(x)
        SHAInet::Profile.stats["ffn.prefill_h2d"]?.should be_nil
        # the host combine is what runs instead
        SHAInet::Profile.stats["ffn.combine"].not_nil![:count].should be > 0
      end
    end
  end

  it "costs one workspace set for many layers, not one per layer" do
    pending! "CUDA/kernels not available" unless moe_prefill_ready?

    # VRAM is the binding limit on context length, and a per-call [rows, d_model]
    # allocation per layer is what this asserts against: at 16k that is 134 MB per
    # layer per call. The claim is a BOUND, so it is asserted as one.
    x = fill!(SHAInet::SimpleMatrix.new(12, 64), 0.4)

    SHAInet::MoEFF.release_prefill_workspaces!
    SHAInet::MoEFF.prefill_workspace_bytes.should eq(0)

    one_layer = 0_u64
    with_moe_device(true) do
      build_moe(64, 96, 6, 2).forward(x)
      one_layer = SHAInet::MoEFF.prefill_workspace_bytes
      one_layer.should be > 0 # the buffers really were used

      # Four more "layers" at the same shape must add NOTHING.
      4.times { build_moe(64, 96, 6, 2).forward(x) }
    end

    SHAInet::MoEFF.prefill_workspace_bytes.should eq(one_layer)
    SHAInet::MoEFF.release_prefill_workspaces!
    SHAInet::MoEFF.prefill_workspace_bytes.should eq(0)
  end

  it "stays bounded across many DIFFERENT shapes, which is what leaked" do
    pending! "CUDA/kernels not available" unless moe_prefill_ready?

    # The regression this pins: workspaces were keyed by the EXACT per-call size, and
    # expert routing hands out slice sizes anywhere in 1..tile. One buffer per distinct
    # size per layer reached ~6.5 GB of VRAM never released, seen in the field as an
    # agent whose VRAM climbed 94% -> 100% over two turns and then failed a 19 MB
    # cudaMalloc.
    #
    # Slice shapes are bucketed to powers of two and row shapes are capped, so the
    # footprint must PLATEAU rather than grow with the number of distinct shapes.
    SHAInet::MoEFF.release_prefill_workspaces!
    moe = build_moe(64, 96, 6, 2)

    with_moe_device(true) do
      # Warm every bucket and fill the row-shape cap.
      [3, 5, 9, 17].each do |rows|
        moe.forward(fill!(SHAInet::SimpleMatrix.new(rows, 64), rows * 0.11))
      end
      settled = SHAInet::MoEFF.prefill_workspace_bytes
      settled.should be > 0

      # Twelve more distinct shapes. Keyed by exact size this grows every time; with
      # the bound in place it must not exceed what the biggest shape needs.
      (2..13).each do |rows|
        moe.forward(fill!(SHAInet::SimpleMatrix.new(rows, 64), rows * 0.07))
      end

      after = SHAInet::MoEFF.prefill_workspace_bytes
      # Allow growth only for genuinely larger shapes, not for their NUMBER: 12 more
      # exact-keyed shapes would have multiplied this.
      after.should be <= settled * 2
    end

    SHAInet::MoEFF.release_prefill_workspaces!
    SHAInet::MoEFF.prefill_workspace_bytes.should eq(0)
  end

  it "caps the row-keyed workspaces rather than keeping one per prompt length" do
    pending! "CUDA/kernels not available" unless moe_prefill_ready?

    # Prompt lengths vary every turn and the result is read back at exactly `rows`, so
    # these cannot be bucketed. They are capped instead, and the cap is the assertion.
    SHAInet::MoEFF.release_prefill_workspaces!
    moe = build_moe(64, 96, 6, 2)

    with_moe_device(true) do
      (2..9).each do |rows|
        moe.forward(fill!(SHAInet::SimpleMatrix.new(rows, 64), rows * 0.3))
      end
    end

    # Eight distinct prompt lengths, at most MAX_ROW_SHAPES retained per cache. The
    # two row caches hold [rows, 64] fp32, so bound the total by the cap times the
    # largest shape, times the two caches.
    largest = 9 * 64 * 4
    cap = SHAInet::MoEFF::MAX_ROW_SHAPES
    SHAInet::MoEFF.prefill_workspace_bytes.should be <= (largest * cap * 2 + largest * 8).to_u64

    SHAInet::MoEFF.release_prefill_workspaces!
  end
end
