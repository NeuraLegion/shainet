require "./spec_helper"

# The fp32 host KV mirror is host RAM the device paths never read: 192 KiB per token per
# layer, and a 32k budget on a 48-layer model reserved about 6.4 GB of it. It used to be
# extended with placeholder zeros purely to keep its length in step, and reserved eagerly
# the moment a KV budget was configured.
#
# The claim is therefore "the device path does not touch it at all", which is asserted
# directly on its size. The safety half matters just as much: with the mirror no longer
# extended, a host path that indexed it would read PAST ITS END rather than read stale
# values, so the guard is asserted too.
private def mirror_ready?
  SHAInet::CUDA.fully_available? &&
    SHAInet::CUDA.prefill_attn_kernels_available? &&
    SHAInet::CUDA.kv_f16_kernels_available?
end

private def with_prefill_device(enabled : Bool?, &)
  prev = SHAInet::LlamaBlock.prefill_attn_device_enabled?
  SHAInet::LlamaBlock.prefill_attn_device_enabled = enabled
  begin
    yield
  ensure
    SHAInet::LlamaBlock.prefill_attn_device_enabled = prev
  end
end

private def seeded(rows, cols, seed)
  m = SHAInet::SimpleMatrix.new(rows, cols)
  rows.times { |i| cols.times { |j| m[i, j] = Math.sin(seed + i * 0.31 + j * 0.17) * 0.5 } }
  m
end

private def block_for(d_model = 64, heads = 4, kv_heads = 2, head_dim = 16)
  blk = SHAInet::LlamaBlock.new(d_model, heads, d_model * 2, num_kv_heads: kv_heads, head_dim: head_dim)
  blk.w_q = seeded(d_model, heads * head_dim, 1.0)
  blk.w_k = seeded(d_model, kv_heads * head_dim, 2.0)
  blk.w_v = seeded(d_model, kv_heads * head_dim, 3.0)
  blk.w_o = seeded(heads * head_dim, d_model, 4.0)
  blk.to_gpu!(quantize: true, bits: 4)
  blk
end

describe "host KV mirror" do
  it "is never populated by the device prefill path" do
    pending! "CUDA/kernels not available" unless mirror_ready?

    blk = block_for
    blk.clear_cache!
    with_prefill_device(true) { blk.forward_cached(seeded(8, 64, 0.6)) }

    # Nothing was appended, and yet generation continues correctly, because the position
    # accounting comes from cache_len once the mirror is marked stale. A following
    # single-token step is the observable proof: it attends over the 8 prefilled
    # positions, which only works if the length is still right.
    blk.host_kv_stale?.should be_true
    blk.host_kv_bytes.should eq(0)
    step = with_prefill_device(true) { blk.forward_cached(seeded(1, 64, 2.2)) }
    step.rows.should eq(1)
    step.cols.should eq(64)
    blk.host_kv_bytes.should eq(0)
  end

  it "does not reserve host RAM for a KV budget the device path will not use" do
    pending! "CUDA/kernels not available" unless mirror_ready?

    # Configuring a budget used to reserve positions * head_dim per kv_head per cache
    # immediately. With 4096 positions on this small block that is still thousands of
    # floats per head; on the 30B it was gigabytes.
    blk = block_for
    blk.clear_cache!
    blk.kv_max_context = 4096
    blk.host_kv_bytes.should eq(0)

    with_prefill_device(true) { blk.forward_cached(seeded(8, 64, 0.9)) }
    blk.host_kv_bytes.should eq(0)
  end

  it "still keeps the mirror when the host path is the one running" do
    pending! "CUDA/kernels not available" unless mirror_ready?

    # The mirror is not gone, it is unused: the host staging path still needs it, so
    # forcing that path must still populate it. Otherwise this change would have broken
    # the fallback rather than made it cheaper.
    blk = block_for
    blk.clear_cache!
    with_prefill_device(false) { blk.forward_cached(seeded(8, 64, 0.6)) }

    blk.host_kv_stale?.should be_false
    blk.host_kv_bytes.should be > 0
  end

  it "refuses the host path after a device prefill instead of reading past the end" do
    pending! "CUDA/kernels not available" unless mirror_ready?

    # This is the safety assertion. The mirror is no longer extended, so a host path that
    # indexed it would run off the end of the array. It must raise instead.
    blk = block_for
    blk.clear_cache!
    with_prefill_device(true) { blk.forward_cached(seeded(6, 64, 0.2)) }

    expect_raises(RuntimeError, /clear_cache!/) do
      with_prefill_device(false) { blk.forward_cached(seeded(4, 64, 0.8)) }
    end

    # And clearing is the documented way back, as the message says.
    blk.clear_cache!
    with_prefill_device(false) { blk.forward_cached(seeded(4, 64, 0.8)) }
    blk.host_kv_stale?.should be_false
  end
end
