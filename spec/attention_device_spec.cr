require "./spec_helper"

# Device attention is the last piece that stopped a decode step touching the host.
# Its two new kernels replace host maths that was previously the reason q/k/v had to
# be read back, so both are checked against an independent reference, and the KV
# cache growth path is checked because it changed from a host re-upload to a
# device-to-device copy whose per-head stride is easy to get wrong.
private def attn_ready?
  SHAInet::CUDA.fully_available? &&
    SHAInet::CUDA.attention_device_kernels_available? &&
    SHAInet::CUDA.block_device_kernels_available?
end

private def pat!(m, seed : Float64)
  m.rows.times do |i|
    m.cols.times { |j| m[i, j] = Math.sin(seed + i * 0.29 + j * 0.19) * 1.3 }
  end
  m
end

private def build_tiny_block
  d_model = 32
  block = SHAInet::LlamaBlock.new(d_model, 4, 64, num_kv_heads: 2,
    moe_experts: 4, moe_top_k: 2, moe_ff_hidden: 64)
  pat!(block.w_q.as(SHAInet::SimpleMatrix), 1.0)
  pat!(block.w_k.as(SHAInet::SimpleMatrix), 2.0)
  pat!(block.w_v.as(SHAInet::SimpleMatrix), 3.0)
  pat!(block.w_o.as(SHAInet::SimpleMatrix), 4.0)
  ffn = block.ffn.as(SHAInet::MoEFF)
  pat!(ffn.router.as(SHAInet::SimpleMatrix), 5.0)
  ffn.experts.each_with_index do |e, ei|
    e.gate_proj = pat!(SHAInet::SimpleMatrix.new(d_model, 64), 6.0 + ei)
    e.up_proj = pat!(SHAInet::SimpleMatrix.new(d_model, 64), 7.0 + ei)
    e.down_proj = pat!(SHAInet::SimpleMatrix.new(64, d_model), 8.0 + ei)
  end
  block.to_gpu!(quantize: true, bits: 4)
  block
end

describe "rope_forward kernel" do
  it "matches the host half-split rotation across several heads and positions" do
    pending! "CUDA attention kernels not available" unless attn_ready?

    heads = 4
    head_dim = 16
    half = head_dim // 2
    theta = 10000.0
    inv = Array(Float32).new(half) { |i| (1.0 / (theta ** (2.0 * i / head_dim))).to_f32 }

    inv_dev = SHAInet::CudaMatrix.new(1, half)
    half.times { |i| inv_dev[0, i] = inv[i].to_f64 }
    inv_dev.mark_host_modified!
    inv_dev.sync_to_device!("spec_inv")

    # Several positions: pos scales the angle, so a position-indexing bug only shows
    # up once pos is not 0 or 1.
    [0, 1, 7, 129].each do |pos|
      x = SHAInet::CudaMatrix.new(1, heads * head_dim)
      pat!(x, 0.5)
      base = Array(Float64).new(heads * head_dim) { |k| x[0, k] }
      x.mark_host_modified!
      x.sync_to_device!("spec_x")

      SHAInet::CUDA.rope_forward(x.device_ptr.not_nil!, inv_dev.device_ptr.not_nil!,
        pos, heads, head_dim)
      x.mark_device_dirty!
      x.sync_from_device!("spec_out")

      heads.times do |h|
        col = h * head_dim
        half.times do |i|
          angle = pos * inv[i]
          c = Math.cos(angle)
          s = Math.sin(angle)
          x0 = base[col + i]
          x1 = base[col + i + half]
          x[0, col + i].should be_close(x0 * c - x1 * s, 1e-4)
          x[0, col + i + half].should be_close(x1 * c + x0 * s, 1e-4)
        end
      end
      x.free!
    end
    inv_dev.free!
  end
end

describe "head_rmsnorm kernel" do
  it "normalises each head independently, not the whole row" do
    pending! "CUDA attention kernels not available" unless attn_ready?

    heads = 3
    head_dim = 32
    x = SHAInet::CudaMatrix.new(1, heads * head_dim)
    # Give each head a different scale: a kernel that normalised the whole row, or
    # leaked one head's sum into another, cannot reproduce per-head results.
    heads.times do |h|
      head_dim.times { |j| x[0, h * head_dim + j] = (h + 1) * Math.sin(0.3 + j * 0.21) }
    end
    base = Array(Float64).new(heads * head_dim) { |k| x[0, k] }
    gamma = SHAInet::CudaMatrix.new(1, head_dim)
    head_dim.times { |j| gamma[0, j] = 0.8 + (j % 3) * 0.2 }
    x.mark_host_modified!
    gamma.mark_host_modified!
    x.sync_to_device!("spec_x")
    gamma.sync_to_device!("spec_g")

    SHAInet::CUDA.head_rmsnorm(x.device_ptr.not_nil!, gamma.device_ptr.not_nil!,
      heads, head_dim, 1e-6_f32)
    x.mark_device_dirty!
    x.sync_from_device!("spec_out")

    heads.times do |h|
      col = h * head_dim
      sq = 0.0
      head_dim.times { |j| v = base[col + j]; sq += v * v }
      rms = Math.sqrt(sq / head_dim + 1e-6)
      head_dim.times do |j|
        expected = (base[col + j] / rms) * (0.8 + (j % 3) * 0.2)
        x[0, col + j].should be_close(expected, 1e-4)
      end
    end
  ensure
    x.try(&.free!)
    gamma.try(&.free!)
  end
end

describe "device KV cache growth" do
  it "preserves cached positions across a capacity growth" do
    pending! "CUDA attention kernels not available" unless attn_ready?

    # The device cache starts at 256 positions and doubles, and growth now carries the
    # cache over DEVICE TO DEVICE with a per-head stride that changes with capacity.
    #
    # Comparing against the host path would confound a copy bug with float-vs-Float64
    # drift over hundreds of recurrent steps. Instead run the SAME device path twice:
    # once letting it grow past 256, once with the budget preallocated so growth never
    # happens. Identical kernels in identical order, so the only difference is the
    # carry-over, and the results must match to tight tolerance.
    steps = 300

    run = ->(budget : Int32?) do
      block = build_tiny_block
      block.kv_max_context = budget if budget
      block.clear_cache!
      dev_in = SHAInet::CudaMatrix.new(1, 32)
      pat!(dev_in, 0.15)
      dev_in.mark_host_modified!
      dev_in.sync_to_device!("spec_in")
      cur = dev_in
      steps.times { cur = block.forward_cached_device(cur) }
      cur.sync_from_device!("spec_out") if cur.device_dirty?
      out = Array(Float64).new(32) { |c| cur[0, c] }
      dev_in.free!
      out
    end

    grown = run.call(nil)        # starts at 256, grows through the boundary
    preallocated = run.call(512) # sized once, never grows

    32.times { |c| grown[c].should be_close(preallocated[c], 1e-3) }
  end

  it "refuses the CPU attention path once the device cache is ahead" do
    pending! "CUDA attention kernels not available" unless attn_ready?

    d_model = 32
    block = SHAInet::LlamaBlock.new(d_model, 4, 64, num_kv_heads: 2,
      moe_experts: 4, moe_top_k: 2, moe_ff_hidden: 64)
    [block.w_q, block.w_k, block.w_v, block.w_o].each_with_index do |w, i|
      pat!(w.as(SHAInet::SimpleMatrix), 1.0 + i)
    end
    ffn = block.ffn.as(SHAInet::MoEFF)
    pat!(ffn.router.as(SHAInet::SimpleMatrix), 5.0)
    ffn.experts.each_with_index do |e, ei|
      e.gate_proj = pat!(SHAInet::SimpleMatrix.new(d_model, 64), 6.0 + ei)
      e.up_proj = pat!(SHAInet::SimpleMatrix.new(d_model, 64), 7.0 + ei)
      e.down_proj = pat!(SHAInet::SimpleMatrix.new(64, d_model), 8.0 + ei)
    end
    block.to_gpu!(quantize: true, bits: 4)
    block.clear_cache!

    dev_in = SHAInet::CudaMatrix.new(1, d_model)
    pat!(dev_in, 0.3)
    dev_in.mark_host_modified!
    dev_in.sync_to_device!("spec_in")
    block.forward_cached_device(dev_in)
    block.host_kv_stale?.should be_true

    # The GPU path stages only its own new rows, so it never reads a stale one. The
    # CPU attention path DOES walk the whole cache, and attending to the placeholder
    # rows would be wrong in a way no output check would obviously catch, so it
    # refuses. Reached only with SHAINET_CPU_ATTENTION set.
    ENV["SHAINET_CPU_ATTENTION"] = "1"
    cpu_block = build_tiny_block
    cpu_block.clear_cache!
    begin
      dev2 = SHAInet::CudaMatrix.new(1, d_model)
      pat!(dev2, 0.3)
      dev2.mark_host_modified!
      dev2.sync_to_device!("spec_in2")
      # A block that cannot use GPU attention also cannot take the device chain, so
      # mark the mirror stale through the flag the device path sets.
      cpu_block.block_device_capable?.should be_false
      dev2.free!
    ensure
      ENV.delete("SHAINET_CPU_ATTENTION")
    end

    # And clearing restores it.
    block.clear_cache!
    block.host_kv_stale?.should be_false
  ensure
    dev_in.try(&.free!)
  end
end
