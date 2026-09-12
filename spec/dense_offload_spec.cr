require "./spec_helper"

# Dense weight offload's claim is a VRAM BOUND, not a speedup: the attention
# projections, the dense FFN and the lm_head stop occupying VRAM and live as
# packed Q4 in host RAM, streamed on demand. Parity alone would not catch a
# regression that quietly left them device-resident, so the bound is asserted
# directly on device_bytes, and the resident figure is asserted FIRST so this
# spec fails loudly if the shapes are ever shrunk into vacuity.
private def offload_ready?
  SHAInet::CUDA.fully_available?
end

# The hot cache's default budget is 70% of whatever VRAM was free when the first
# host weight was touched, memoized process-globally. Left at that, these examples
# are the first thing in the suite to touch it and every later example inherits a
# cache that may hold most of the card. Bound it here and release it after, so
# these examples neither depend on nor disturb ambient state.
private def with_bounded_cache(&)
  SHAInet::Q4HostMatrix.budget_bytes = 32_u64 * 1024 * 1024
  begin
    yield
  ensure
    SHAInet::Q4HostMatrix.release_cache!
  end
end

private def fill!(m, seed : Float64)
  m.rows.times do |i|
    m.cols.times { |j| m[i, j] = Math.sin(seed + i * 0.29 + j * 0.17) * 0.5 }
  end
  m
end

private def build_block(d_model : Int32 = 64, heads : Int32 = 4, ff : Int32 = 128)
  b = SHAInet::LlamaBlock.new(d_model, heads, ff)
  fill!(b.w_q.as(SHAInet::SimpleMatrix), 0.11)
  fill!(b.w_k.as(SHAInet::SimpleMatrix), 0.23)
  fill!(b.w_v.as(SHAInet::SimpleMatrix), 0.37)
  fill!(b.w_o.as(SHAInet::SimpleMatrix), 0.51)
  ffn = b.ffn.as(SHAInet::SwiGLUFF)
  fill!(ffn.gate_proj.as(SHAInet::SimpleMatrix), 0.61)
  fill!(ffn.up_proj.as(SHAInet::SimpleMatrix), 0.71)
  fill!(ffn.down_proj.as(SHAInet::SimpleMatrix), 0.83)
  b
end

private def proj_device_bytes(b) : UInt64
  [b.w_q, b.w_k, b.w_v, b.w_o].sum do |w|
    w.is_a?(SHAInet::QuantizedWeight) ? w.device_bytes : 0_u64
  end
end

private def tiny_net
  net = SHAInet::Network.new
  net.add_layer(:embedding, 32, vocab_size: 48)
  net.add_layer(:llama, 32, num_heads: 4, ff_hidden: 64)
  net.add_layer(:output, 48, activation_function: SHAInet.identity)
  net.fully_connect
  net
end

describe "dense weight offload" do
  it "keeps the attention projections in VRAM without offload" do
    pending! "CUDA not available" unless offload_ready?

    with_bounded_cache do
      b = build_block
      b.to_gpu!(quantize: true, bits: 4)

      # The floor is DERIVED from the shapes via the library's own accounting
      # rather than hardcoded, so it tracks the block instead of drifting. All
      # four projections are 64x64 here (head_dim 16 * 4 heads = q_dim 64).
      expected = 4_u64 * SHAInet::Q4CudaMatrix.device_bytes_for(64, 64)
      resident = proj_device_bytes(b)
      resident.should eq(expected)
      # Guard against the assertion going vacuous if the block ever shrinks: a
      # real Q4 block of this size is several KiB, not a handful of bytes.
      resident.should be > 8_192_u64

      [b.w_q, b.w_k, b.w_v, b.w_o].each(&.should(be_a(SHAInet::Q4CudaMatrix)))
    end
  end

  it "moves the attention projections out of VRAM entirely with offload" do
    pending! "CUDA not available" unless offload_ready?

    with_bounded_cache do
      b = build_block
      b.to_gpu!(quantize: true, bits: 4, offload: true)

      # The whole point: zero resident VRAM for these weights.
      proj_device_bytes(b).should eq(0_u64)

      [b.w_q, b.w_k, b.w_v, b.w_o].each do |w|
        w.should be_a(SHAInet::Q4HostMatrix)
        # and the bytes really exist, in host RAM, rather than having been dropped.
        w.as(SHAInet::Q4HostMatrix).host_bytes.should be > 0_u64
      end
    end
  end

  it "offloads the dense FFN as well as attention" do
    pending! "CUDA not available" unless offload_ready?

    with_bounded_cache do
      b = build_block
      b.to_gpu!(quantize: true, bits: 4, offload: true)
      ffn = b.ffn.as(SHAInet::SwiGLUFF)

      [ffn.gate_proj, ffn.up_proj, ffn.down_proj].each do |w|
        w.should be_a(SHAInet::Q4HostMatrix)
        w.as(SHAInet::QuantizedWeight).device_bytes.should eq(0_u64)
      end
    end
  end

  it "produces the same numbers offloaded as resident" do
    pending! "CUDA not available" unless offload_ready?

    with_bounded_cache do
      # Same Q4 packing and same kernel in both cases: only residency differs, so
      # the projection results must agree, not merely fall inside a wide band.
      resident = build_block
      offloaded = build_block
      resident.to_gpu!(quantize: true, bits: 4)
      offloaded.to_gpu!(quantize: true, bits: 4, offload: true)

      x = SHAInet::CudaMatrix.new(1, 64)
      64.times { |j| x[0, j] = Math.cos(j * 0.21) * 0.3 }
      x.mark_host_modified!
      x.sync_to_device!("dense_offload_spec_in")

      a = resident.w_q.as(SHAInet::QuantizedWeight).gemv(x)
      c = offloaded.w_q.as(SHAInet::QuantizedWeight).gemv(x)
      a.sync_from_device!("resident_out") if a.device_dirty?
      c.sync_from_device!("offloaded_out") if c.device_dirty?

      a.cols.should eq(c.cols)
      a.cols.times { |j| c[0, j].should be_close(a[0, j], 1e-6) }
    end
  end

  it "still computes correctly with the hot cache disabled" do
    pending! "CUDA not available" unless offload_ready?

    # Budget 0 disables caching, so every GEMV streams through the shared
    # scratch. That is the worst-case capacity mode and must still be correct.
    SHAInet::Q4HostMatrix.budget_bytes = 0_u64
    begin
      resident = build_block
      streamed = build_block
      resident.to_gpu!(quantize: true, bits: 4)
      streamed.to_gpu!(quantize: true, bits: 4, offload: true)

      x = SHAInet::CudaMatrix.new(1, 64)
      64.times { |j| x[0, j] = Math.sin(j * 0.17) * 0.4 }
      x.mark_host_modified!
      x.sync_to_device!("dense_offload_nocache_in")

      a = resident.w_q.as(SHAInet::QuantizedWeight).gemv(x)
      c = streamed.w_q.as(SHAInet::QuantizedWeight).gemv(x)
      a.sync_from_device!("nocache_resident") if a.device_dirty?
      c.sync_from_device!("nocache_streamed") if c.device_dirty?
      a.cols.times { |j| c[0, j].should be_close(a[0, j], 1e-6) }
    ensure
      SHAInet::Q4HostMatrix.release_cache!
    end
  end

  it "refuses offload at 8 bits instead of silently ignoring it" do
    pending! "CUDA not available" unless offload_ready?

    with_bounded_cache do
      b = build_block
      expect_raises(ArgumentError, /4-bit/) do
        b.to_gpu!(quantize: true, bits: 8, offload: true)
      end
    end
  end

  it "refuses offload without quantization" do
    pending! "CUDA not available" unless offload_ready?

    with_bounded_cache do
      b = build_block
      expect_raises(ArgumentError, /requires quantization/) do
        b.to_gpu!(quantize: false, bits: 4, offload: true)
      end
    end
  end

  it "moves the lm_head off the device too" do
    pending! "CUDA not available" unless offload_ready?

    with_bounded_cache do
      # On a real vocabulary the lm_head is the single biggest dense tensor
      # (Qwen3's 151936-wide head is ~1.2 GB in fp32), so it is the highest-value
      # one to move. Asserted on residency, not just on type.
      net = tiny_net
      net.quantize!(4, offload: true)

      head = net.lm_head_q
      head.should_not be_nil
      head.not_nil!.should be_a(SHAInet::Q4HostMatrix)
      head.not_nil!.device_bytes.should eq(0_u64)
    end
  end

  it "keeps the lm_head in VRAM without offload" do
    pending! "CUDA not available" unless offload_ready?

    with_bounded_cache do
      net = tiny_net
      net.quantize!(4)

      head = net.lm_head_q
      head.should_not be_nil
      head.not_nil!.should be_a(SHAInet::Q4CudaMatrix)
      head.not_nil!.device_bytes.should be > 0_u64
    end
  end

  it "refuses lm_head offload at 8 bits" do
    pending! "CUDA not available" unless offload_ready?

    with_bounded_cache do
      net = tiny_net
      expect_raises(ArgumentError, /4-bit/) { net.quantize!(8, offload: true) }
    end
  end
end
