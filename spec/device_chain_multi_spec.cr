require "./spec_helper"

# Multi-token device block chain. With the FFN and attention each device-resident but the
# chain still host-driven, every layer paid an upload and a readback around each of them:
# attn.d2h alone measured 40.7% of a 4096-token prefill on a 30B-A3B.
#
# Two claims, and the order matters. Parity first, because a chain that is fast and wrong
# is worse than the host path. Then the transfer budget, asserted as a COUNT over a stack
# of layers, since "one per stack" and "one per layer" are indistinguishable with a single
# layer.
private def chain_ready?
  SHAInet::CUDA.fully_available? &&
    SHAInet::CUDA.prefill_attn_kernels_available? &&
    SHAInet::CUDA.kv_f16_kernels_available? &&
    SHAInet::CUDA.swiglu_kernel_available?
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

private def fill!(m, seed)
  m.rows.times { |i| m.cols.times { |j| m[i, j] = Math.sin(seed + i * 0.23 + j * 0.11) * 0.5 } }
  m
end

private def stack_of(layers : Int32, d_model = 64, heads = 4, kv_heads = 2, head_dim = 16)
  Array(SHAInet::LlamaBlock).new(layers) do |idx|
    b = SHAInet::LlamaBlock.new(d_model, heads, d_model * 2,
      num_kv_heads: kv_heads, head_dim: head_dim)
    b.w_q = fill!(SHAInet::SimpleMatrix.new(d_model, heads * head_dim), 1.0 + idx)
    b.w_k = fill!(SHAInet::SimpleMatrix.new(d_model, kv_heads * head_dim), 2.0 + idx)
    b.w_v = fill!(SHAInet::SimpleMatrix.new(d_model, kv_heads * head_dim), 3.0 + idx)
    b.w_o = fill!(SHAInet::SimpleMatrix.new(heads * head_dim, d_model), 4.0 + idx)
    b.to_gpu!(quantize: true, bits: 4)
    b.clear_cache!
    b
  end
end

describe "multi-token device block chain" do
  it "matches the host chain over a stack of layers" do
    pending! "CUDA/kernels not available" unless chain_ready?

    d_model = 64
    x = fill!(SHAInet::SimpleMatrix.new(7, d_model), 0.6)

    host = begin
      cur = x
      with_prefill_device(false) do
        stack_of(3).each { |b| cur = b.forward_cached(cur) }
      end
      cur
    end

    dev = begin
      blocks = stack_of(3)
      gx = x.to_cuda
      gx.sync_to_device!("spec_chain_in") unless gx.device_dirty?
      cur = gx
      out = nil
      with_prefill_device(true) do
        blocks.each do |b|
          nxt = b.forward_cached_device_multi(cur)
          nxt.should_not be_nil # declining here would make the parity below vacuous
          cur = nxt.not_nil!
        end
        cur.sync_from_device!("spec_chain_out") if cur.device_dirty?
        out = cur.to_simple
      end
      out.not_nil!
    end

    host.rows.should eq(dev.rows)
    host.cols.should eq(dev.cols)
    host.rows.times do |i|
      host.cols.times { |j| dev[i, j].should be_close(host[i, j], 5e-3) }
    end
  end

  it "pays no per-layer host round trip across the stack" do
    pending! "CUDA/kernels not available" unless chain_ready?

    # The budget, asserted as an ABSENCE over THREE layers: the phases that mark a host
    # round trip inside a block must not fire at all. With one layer this assertion could
    # not tell "once per stack" from "once per layer".
    blocks = stack_of(3)
    x = fill!(SHAInet::SimpleMatrix.new(7, 64), 1.4)

    with_prefill_device(true) do
      with_prof do
        gx = x.to_cuda
        gx.sync_to_device!("spec_chain_in") unless gx.device_dirty?
        cur = gx
        blocks.each { |b| cur = b.forward_cached_device_multi(cur).not_nil! }

        stats = SHAInet::Profile.stats
        # Attention's output readback is the one this exists to remove.
        stats["attn.d2h"]?.should be_nil
        # The FFN's transfers either side of it, likewise.
        stats["ffn.prefill_h2d"]?.should be_nil
        stats["ffn.prefill_d2h"]?.should be_nil
        # And the work really happened, three layers' worth, so this is not vacuous.
        stats["block.dev_norm"].not_nil![:count].should eq(6)
        stats["attn.dev_wo"].not_nil![:count].should eq(3)
      end
    end
  end

  it "gives the same answer chunked as unchunked" do
    pending! "CUDA/kernels not available" unless chain_ready?

    # Chunking is only sound because the block is causal and per-token everywhere it is
    # not: chunk B's attention sees chunk A's KV because A was appended first, and the
    # norms, FFN and residuals treat tokens independently. That argument is what this
    # asserts, over a prompt split into four chunks against the same prompt in one.
    x = fill!(SHAInet::SimpleMatrix.new(12, 64), 0.77)

    run = ->(chunk : Int32?) do
      prev = SHAInet::LlamaBlock.prefill_chunk
      SHAInet::LlamaBlock.prefill_chunk = chunk
      begin
        blocks = stack_of(3)
        gx = x.to_cuda
        gx.sync_to_device!("spec_chunk_in") unless gx.device_dirty?
        cur = gx
        result = nil
        with_prefill_device(true) do
          blocks.each do |b|
            nxt = b.forward_cached_device_multi(cur)
            nxt.should_not be_nil
            cur = nxt.not_nil!
          end
          cur.sync_from_device!("spec_chunk_out") if cur.device_dirty?
          result = cur.to_simple
        end
        result.not_nil!
      ensure
        SHAInet::LlamaBlock.prefill_chunk = prev
      end
    end

    whole = run.call(64) # one chunk covers all 12 rows
    split = run.call(3)  # four chunks of three

    whole.rows.times do |i|
      whole.cols.times { |j| split[i, j].should be_close(whole[i, j], 5e-3) }
    end
  end

  it "stops scaling workspace VRAM with prompt length" do
    pending! "CUDA/kernels not available" unless chain_ready?

    # The measured failure this fixes: a 20480-token prefill died on a 160 MB cudaMalloc,
    # and 167772160 / 4 / 20480 is exactly d_model, so it was a full-prompt-length
    # workspace. Chunked, a longer prompt must NOT enlarge them.
    prev = SHAInet::LlamaBlock.prefill_chunk
    SHAInet::LlamaBlock.prefill_chunk = 4
    begin
      short = 0_u64
      long = 0_u64

      SHAInet::LlamaBlock.release_attn_workspaces!
      with_prefill_device(true) do
        b = stack_of(1).first
        b.forward_cached_device_multi(fill!(SHAInet::SimpleMatrix.new(8, 64), 0.3).to_cuda)
        short = SHAInet::LlamaBlock.attn_workspace_bytes
      end
      short.should be > 0

      SHAInet::LlamaBlock.release_attn_workspaces!
      with_prefill_device(true) do
        b = stack_of(1).first
        # Five times the tokens, same chunk size.
        b.forward_cached_device_multi(fill!(SHAInet::SimpleMatrix.new(40, 64), 0.3).to_cuda)
        long = SHAInet::LlamaBlock.attn_workspace_bytes
      end

      # Identical, not merely similar: every workspace is sized to the chunk, so a 5x
      # longer prompt adds nothing.
      long.should eq(short)
    ensure
      SHAInet::LlamaBlock.prefill_chunk = prev
      SHAInet::LlamaBlock.release_attn_workspaces!
    end
  end
end
