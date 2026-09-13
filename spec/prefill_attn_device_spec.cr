require "./spec_helper"

# Device-resident PREFILL attention against the host path it replaces.
#
# This is the assertion that matters: the change moves bias, QK-norm, RoPE and the KV
# append onto the device, and any ordering or layout mistake produces output that still
# LOOKS like attention. A transfer-count assertion would pass while the numbers were
# wrong, so parity is checked first and the transfer budget second.
private def prefill_attn_ready?
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
  rows.times { |i| cols.times { |j| m[i, j] = Math.sin(seed + i * 0.29 + j * 0.13) * 0.5 } }
  m
end

# Runs the block on the device path and PROVES it was taken. Without this a parity
# example passes vacuously whenever the device path declines and both sides run the same
# host code -- which is the most likely way this spec would silently stop testing
# anything.
private def run_on_device(&) : SHAInet::SimpleMatrix
  prev = SHAInet::Profile.enabled?
  SHAInet::Profile.enabled = true
  SHAInet::Profile.reset
  begin
    result = with_prefill_device(true) { yield }
    SHAInet::Profile.stats["attn.dev_pack"]?.should_not be_nil
    result
  ensure
    SHAInet::Profile.enabled = prev
    SHAInet::Profile.reset
  end
end

# A block small enough to be quick but with the shape that matters: q_dim != d_model,
# and grouped-query attention (num_heads > num_kv_heads), which is what the 30B has.
private def build_block(d_model : Int32, heads : Int32, kv_heads : Int32, head_dim : Int32)
  blk = SHAInet::LlamaBlock.new(d_model, heads, d_model * 2, num_kv_heads: kv_heads, head_dim: head_dim)
  blk.w_q = seeded(d_model, heads * head_dim, 1.0)
  blk.w_k = seeded(d_model, kv_heads * head_dim, 2.0)
  blk.w_v = seeded(d_model, kv_heads * head_dim, 3.0)
  blk.w_o = seeded(heads * head_dim, d_model, 4.0)
  blk
end

describe "device-resident prefill attention" do
  it "matches the host path over a multi-token prompt" do
    pending! "CUDA/kernels not available" unless prefill_attn_ready?

    d_model = 64
    heads = 4
    kv_heads = 2
    head_dim = 16 # q_dim = 64 here; the 30B's q_dim != d_model case is covered below
    x = seeded(8, d_model, 0.7)

    host = with_prefill_device(false) do
      b = build_block(d_model, heads, kv_heads, head_dim)
      b.to_gpu!(quantize: true, bits: 4)
      b.clear_cache!
      b.forward_cached(x)
    end

    dev = run_on_device do
      b = build_block(d_model, heads, kv_heads, head_dim)
      b.to_gpu!(quantize: true, bits: 4)
      b.clear_cache!
      b.forward_cached(x)
    end

    host.rows.should eq(dev.rows)
    host.cols.should eq(dev.cols)
    host.rows.times do |i|
      host.cols.times { |j| dev[i, j].should be_close(host[i, j], 5e-3) }
    end
  end

  it "matches the host path when q_dim is larger than d_model" do
    pending! "CUDA/kernels not available" unless prefill_attn_ready?

    # The trap this pins: on Qwen3-Coder-30B-A3B q_dim is 32 * 128 = 4096 while d_model is
    # 2048, so the attention output stride is q_dim, not d_model. A stride mistake is
    # silently wrong rather than a crash, so the geometry is reproduced here in miniature.
    d_model = 32
    heads = 4
    kv_heads = 2
    head_dim = 16 # q_dim = 64 = 2 * d_model
    x = seeded(6, d_model, 1.9)

    host = with_prefill_device(false) do
      b = build_block(d_model, heads, kv_heads, head_dim)
      b.to_gpu!(quantize: true, bits: 4)
      b.clear_cache!
      b.forward_cached(x)
    end

    dev = run_on_device do
      b = build_block(d_model, heads, kv_heads, head_dim)
      b.to_gpu!(quantize: true, bits: 4)
      b.clear_cache!
      b.forward_cached(x)
    end

    host.rows.times do |i|
      host.cols.times { |j| dev[i, j].should be_close(host[i, j], 5e-3) }
    end
  end

  it "keeps decoding correctly after a device prefill" do
    pending! "CUDA/kernels not available" unless prefill_attn_ready?

    # The device path fills the host KV mirror with placeholder zeros and marks it stale,
    # so a following single-token step must still attend to the real device-side KV. If
    # the length parity or the staleness flag were wrong this is where it shows.
    d_model = 64
    heads = 4
    kv_heads = 2
    head_dim = 16
    prompt = seeded(7, d_model, 0.4)
    step = seeded(1, d_model, 2.7)

    host = with_prefill_device(false) do
      b = build_block(d_model, heads, kv_heads, head_dim)
      b.to_gpu!(quantize: true, bits: 4)
      b.clear_cache!
      b.forward_cached(prompt)
      b.forward_cached(step)
    end

    dev = run_on_device do
      b = build_block(d_model, heads, kv_heads, head_dim)
      b.to_gpu!(quantize: true, bits: 4)
      b.clear_cache!
      b.forward_cached(prompt)
      b.forward_cached(step)
    end

    host.rows.times do |i|
      host.cols.times { |j| dev[i, j].should be_close(host[i, j], 5e-3) }
    end
  end

  it "stops staging KV through the host" do
    pending! "CUDA/kernels not available" unless prefill_attn_ready?

    # The performance claim, asserted as an ABSENCE: the host no longer rebuilds a KV blob
    # from its mirror and uploads it. attn.dev_pack replaces attn.stage_host + attn.h2d.
    d_model = 64
    x = seeded(8, d_model, 1.3)

    prev = SHAInet::Profile.enabled?
    SHAInet::Profile.enabled = true
    SHAInet::Profile.reset
    begin
      with_prefill_device(true) do
        b = build_block(d_model, 4, 2, 16)
        b.to_gpu!(quantize: true, bits: 4)
        b.clear_cache!
        b.forward_cached(x)
      end

      stats = SHAInet::Profile.stats
      stats["attn.stage_host"]?.should be_nil
      stats["attn.h2d"]?.should be_nil
      stats["attn.dev_pack"].not_nil![:count].should be > 0
      # And the projections were not read back to be re-uploaded.
      stats["attn.dev_qkv"].not_nil![:count].should be > 0
    ensure
      SHAInet::Profile.enabled = prev
      SHAInet::Profile.reset
    end
  end

  it "shares attention workspaces across layers instead of one set per layer" do
    pending! "CUDA/kernels not available" unless prefill_attn_ready?

    # Measured regression this pins: these buffers are ~117 MB per layer at a 4096-token
    # prefill (q alone is 4096 x 4096 x 4), so a set per layer is ~5.6 GB across 48. Held
    # per-instance it pushed a 30B to 15882 MB of 16376 and the next prefill failed a
    # 64 MB allocation. Only one layer runs at a time, so N layers must cost ONE set.
    d_model = 64
    x = seeded(8, d_model, 0.55)

    SHAInet::LlamaBlock.release_attn_workspaces!
    SHAInet::LlamaBlock.attn_workspace_bytes.should eq(0)

    one = 0_u64
    with_prefill_device(true) do
      b = build_block(d_model, 4, 2, 16)
      b.to_gpu!(quantize: true, bits: 4)
      b.clear_cache!
      b.forward_cached(x)
      one = SHAInet::LlamaBlock.attn_workspace_bytes
      one.should be > 0 # really allocated, so this is not vacuous

      # Four more layers at the same prompt shape must add nothing.
      4.times do
        blk = build_block(d_model, 4, 2, 16)
        blk.to_gpu!(quantize: true, bits: 4)
        blk.clear_cache!
        blk.forward_cached(x)
      end
    end

    SHAInet::LlamaBlock.attn_workspace_bytes.should eq(one)
    SHAInet::LlamaBlock.release_attn_workspaces!
    SHAInet::LlamaBlock.attn_workspace_bytes.should eq(0)
  end
end
