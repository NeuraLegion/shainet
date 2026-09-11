require "./spec_helper"

# The headline claim of the device-resident FFN is a TRANSFER BUDGET: one readback
# per token instead of three per expert. Parity alone would not catch a regression
# that quietly reintroduces the readbacks, so the count is asserted directly, and
# asserted in the other direction on the host path with the SAME weights.
private def cuda_ready?
  SHAInet::CUDA.fully_available? && SHAInet::CUDA.swiglu_kernel_available?
end

private def with_device_decode(enabled : Bool?, &)
  prev = SHAInet::MoEFF.device_decode_enabled?
  SHAInet::MoEFF.device_decode_enabled = enabled
  begin
    yield
  ensure
    SHAInet::MoEFF.device_decode_enabled = prev
  end
end

private def with_profile(&)
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

# Deterministic weights: a fixed pattern, not RNG, so a failure is reproducible
# and does not depend on suite ordering seeding Random::DEFAULT.
private def fill_pattern!(m : SHAInet::SimpleMatrix, seed : Float64)
  m.rows.times do |i|
    m.cols.times do |j|
      m[i, j] = Math.sin(seed + i * 0.37 + j * 0.11) * 0.5
    end
  end
  m
end

private def build_moe(d_model : Int32, ff_hidden : Int32, num_experts : Int32, top_k : Int32)
  moe = SHAInet::MoEFF.new(d_model, ff_hidden, num_experts, top_k)
  fill_pattern!(moe.router.as(SHAInet::SimpleMatrix), 0.9)
  moe.experts.each_with_index do |e, idx|
    e.gate_proj = fill_pattern!(SHAInet::SimpleMatrix.new(d_model, ff_hidden), 1.0 + idx)
    e.up_proj = fill_pattern!(SHAInet::SimpleMatrix.new(d_model, ff_hidden), 2.0 + idx)
    e.down_proj = fill_pattern!(SHAInet::SimpleMatrix.new(ff_hidden, d_model), 3.0 + idx)
  end
  moe
end

describe "device-resident MoE decode" do
  it "matches the host path on a real GPU" do
    pending! "CUDA with the fused SwiGLU kernel not available" unless cuda_ready?

    d_model = 64
    top_k = 4
    moe = build_moe(d_model, 128, 8, top_k)
    x = fill_pattern!(SHAInet::SimpleMatrix.new(1, d_model), 0.25)

    # Quantize once, then run BOTH paths against the identical quantized weights so
    # the comparison isolates the plumbing rather than mixing in quantization error.
    moe.to_gpu!(quantize: true, bits: 4)

    host = with_device_decode(false) { moe.forward(x) }
    dev = with_device_decode(true) { moe.forward(x) }

    dev.rows.should eq host.rows
    dev.cols.should eq host.cols
    d_model.times do |c|
      dev[0, c].should be_close(host[0, c], 1e-3)
    end
  end

  it "spends ONE readback per token instead of three per expert" do
    pending! "CUDA with the fused SwiGLU kernel not available" unless cuda_ready?

    d_model = 64
    top_k = 4
    moe = build_moe(d_model, 128, 8, top_k)
    x = fill_pattern!(SHAInet::SimpleMatrix.new(1, d_model), 0.25)
    moe.to_gpu!(quantize: true, bits: 4)

    device_readbacks = 0
    host_readbacks = 0

    with_profile do
      with_device_decode(true) { moe.forward(x) }
      stats = SHAInet::Profile.stats
      # Proves the fast path actually RAN. Without this the parity spec above could
      # pass by silently falling back to the host path.
      stats["ffn.dev_gate_up"]?.should_not be_nil
      stats["ffn.dev_gate_up"].not_nil![:count].should eq top_k
      device_readbacks = stats["ffn.dev_readback"].not_nil![:count]
    end

    with_profile do
      with_device_decode(false) { moe.forward(x) }
      host_readbacks = SHAInet::Profile.stats["gemm.out_d2h"].not_nil![:count]
    end

    # The budget: one readback for the whole FFN regardless of how many experts ran.
    device_readbacks.should eq 1
    # And the guard in the other direction: the host path really does pay three per
    # expert (gate, up, down), so the saving is not an artifact of the measurement.
    host_readbacks.should eq 3 * top_k
    device_readbacks.should be < host_readbacks
  end

  it "falls back to the host path when experts are not quantized" do
    d_model = 32
    moe = build_moe(d_model, 64, 4, 2)
    x = fill_pattern!(SHAInet::SimpleMatrix.new(1, d_model), 0.4)

    # No to_gpu!, so the experts hold SimpleMatrix weights and the device path must
    # decline rather than raise or produce zeros.
    with_device_decode(true) do
      out = moe.forward(x)
      out.rows.should eq 1
      out.cols.should eq d_model
      out.data.any? { |v| v != 0.0 }.should be_true
    end
  end

  it "declines the device path for multi-token prefill" do
    pending! "CUDA with the fused SwiGLU kernel not available" unless cuda_ready?

    d_model = 32
    moe = build_moe(d_model, 64, 4, 2)
    moe.to_gpu!(quantize: true, bits: 4)
    x = fill_pattern!(SHAInet::SimpleMatrix.new(3, d_model), 0.4)

    with_profile do
      out = with_device_decode(true) { moe.forward(x) }
      out.rows.should eq 3
      # Prefill is a different shape entirely; the single-row device path must not
      # claim it and silently drop rows.
      SHAInet::Profile.stats["ffn.dev_readback"]?.should be_nil
    end
  end
end

describe SHAInet::SwiGLUFF do
  it "forward_device matches forward for one expert" do
    pending! "CUDA with the fused SwiGLU kernel not available" unless cuda_ready?

    d_model = 64
    ff_hidden = 128
    ff = SHAInet::SwiGLUFF.new(d_model, ff_hidden)
    ff.gate_proj = fill_pattern!(SHAInet::SimpleMatrix.new(d_model, ff_hidden), 1.5)
    ff.up_proj = fill_pattern!(SHAInet::SimpleMatrix.new(d_model, ff_hidden), 2.5)
    ff.down_proj = fill_pattern!(SHAInet::SimpleMatrix.new(ff_hidden, d_model), 3.5)
    ff.to_gpu!(quantize: true, bits: 4)
    ff.device_resident_capable?.should be_true

    x = fill_pattern!(SHAInet::SimpleMatrix.new(1, d_model), 0.7)
    host = ff.forward(x)

    xb = SHAInet::CudaMatrix.new(1, d_model)
    xb.raw_data.to_unsafe.copy_from(x.data.to_unsafe, d_model)
    xb.mark_host_modified!
    xb.sync_to_device!("spec_in")
    out_buf = SHAInet::CudaMatrix.new(1, d_model)
    ff.forward_device(xb, out_buf)
    out_buf.sync_from_device!("spec_out")

    d_model.times do |c|
      out_buf[0, c].should be_close(host[0, c], 1e-3)
    end
  end
end

describe "swiglu kernel" do
  it "computes silu(gate) * up against an independent CPU reference" do
    pending! "CUDA with the fused SwiGLU kernel not available" unless cuda_ready?

    n = 1024
    gate = SHAInet::CudaMatrix.new(1, n)
    up = SHAInet::CudaMatrix.new(1, n)
    hidden = SHAInet::CudaMatrix.new(1, n)

    expected = Array(Float64).new(n, 0.0)
    n.times do |i|
      # Span negative, zero and positive, so a sign error or a swapped operand
      # cannot pass: silu is asymmetric and silu(0) is exactly 0.
      g = (i - n // 2) * 0.01
      u = Math.cos(i * 0.05)
      gate[0, i] = g
      up[0, i] = u
      expected[i] = (g / (1.0 + Math.exp(-g))) * u
    end
    gate.mark_host_modified!
    up.mark_host_modified!
    gate.sync_to_device!("spec_gate")
    up.sync_to_device!("spec_up")

    SHAInet::CUDA.swiglu_forward(hidden.device_ptr.not_nil!,
      gate.device_ptr.not_nil!, up.device_ptr.not_nil!, n)
    hidden.mark_device_dirty!
    hidden.sync_from_device!("spec_hidden")

    n.times do |i|
      hidden[0, i].should be_close(expected[i], 1e-5)
    end
  end
end
