require "./spec_helper"

# The whole point of chunked prefill is a MEMORY bound, so it is asserted here
# rather than assumed. Unchunked, the attention workspace grows as
# num_heads * seq^2, which is what made a long prefill impossible. These specs
# are deterministic (allocation sizes, not timing) and must always run.
private def fill_random!(w, rng : Random)
  m = w.as(SHAInet::SimpleMatrix)
  m.rows.times { |r| m.cols.times { |c| m[r, c] = (rng.rand * 0.2 - 0.1) } }
end

private def budget_floats
  SHAInet::LlamaBlock::ATTN_WS_BUDGET_FLOATS
end

private def chunk_max
  SHAInet::LlamaBlock::ATTN_CHUNK_MAX
end

describe "LlamaBlock attention workspace bound" do
  # Pure size policy. This is where the budget clamp is actually guarded: the
  # overshoot only triggers for a specific prior capacity, which an integration
  # spec cannot reliably reproduce, so it is pinned here instead.
  describe ".next_buf_cap" do
    it "allocates exactly what is needed from empty" do
      SHAInet::LlamaBlock.next_buf_cap(0, 100).should eq 100
    end

    it "doubles rather than creeping, to avoid realloc churn" do
      SHAInet::LlamaBlock.next_buf_cap(100, 150).should eq 200
    end

    it "does not reallocate below the current capacity" do
      SHAInet::LlamaBlock.next_buf_cap(500, 100).should eq 1000
    end

    it "clamps the doubling to cap_limit instead of overshooting the budget" do
      limit = budget_floats.to_i32
      # cur_cap just under the limit: doubling would land at 80M, well past a
      # 67.1M budget. The clamp must hold it to the limit.
      SHAInet::LlamaBlock.next_buf_cap(40_000_000, 50_000_000, cap_limit: limit).should eq limit
      # Without the clamp this is the overshoot the clamp exists to prevent.
      SHAInet::LlamaBlock.next_buf_cap(40_000_000, 50_000_000).should eq 80_000_000
    end

    it "still honours a single request larger than cap_limit" do
      limit = budget_floats.to_i32
      # Under-allocating would let a kernel write past the buffer, so `needed`
      # wins over the limit.
      SHAInet::LlamaBlock.next_buf_cap(0, limit + 4096, cap_limit: limit).should eq limit + 4096
    end
  end

  describe "chunk sizing" do
    it "never exceeds ATTN_CHUNK_MAX and never drops below one token" do
      blk = SHAInet::LlamaBlock.new(64, 8, 128, num_kv_heads: 4)
      [1, 16, 256, 1024, 8192, 32_768, 131_072, 1_000_000].each do |total_len|
        n = blk.attn_chunk_tokens(total_len)
        n.should be >= 1
        n.should be <= chunk_max
      end
    end

    it "keeps num_heads * chunk * total_len inside the budget" do
      # 32 query heads is the 30B-class shape where the quadratic term bites.
      blk = SHAInet::LlamaBlock.new(1024, 32, 2048, num_kv_heads: 4, head_dim: 32)
      [2048, 8192, 32_768, 131_072].each do |total_len|
        n = blk.attn_chunk_tokens(total_len)
        ws = 32_i64 * n.to_i64 * total_len.to_i64
        ws.should be <= budget_floats
      end
    end

    it "shrinks the chunk as context grows, which is what bounds the scratch" do
      blk = SHAInet::LlamaBlock.new(1024, 32, 2048, num_kv_heads: 4, head_dim: 32)
      at_8k = blk.attn_chunk_tokens(8192)
      at_128k = blk.attn_chunk_tokens(131_072)
      # A fixed chunk size would let the workspace grow linearly without limit;
      # the sizing must trade chunk width for context depth.
      at_128k.should be < at_8k
    end

    it "honours an explicit SHAINET_ATTN_CHUNK override" do
      blk = SHAInet::LlamaBlock.new(64, 8, 128, num_kv_heads: 4)
      prev = ENV["SHAINET_ATTN_CHUNK"]?
      begin
        ENV["SHAINET_ATTN_CHUNK"] = "7"
        blk.attn_chunk_tokens(32_768).should eq 7
        # A nonsense value must fall back to the computed size, not to zero.
        ENV["SHAINET_ATTN_CHUNK"] = "0"
        blk.attn_chunk_tokens(32_768).should be >= 1
        ENV["SHAINET_ATTN_CHUNK"] = "not-a-number"
        blk.attn_chunk_tokens(32_768).should be >= 1
      ensure
        if prev
          ENV["SHAINET_ATTN_CHUNK"] = prev
        else
          ENV.delete("SHAINET_ATTN_CHUNK")
        end
      end
    end
  end

  describe "allocated workspace" do
    it "stays within the budget after a prefill whose unchunked scratch would not" do
      pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

      # 8 heads at seq 3000 => unchunked scratch is 8 * 3000 * 3000 = 72M floats
      # (288 MB), above the 256 MB budget, so chunking must actually engage and
      # the assertion below is not vacuous.
      heads = 8
      head_dim = 8
      d_model = heads * head_dim
      seq = 3000
      unchunked = heads.to_i64 * seq.to_i64 * seq.to_i64
      unchunked.should be > budget_floats # guard: keep this spec meaningful

      blk = SHAInet::LlamaBlock.new(d_model, heads, 64, num_kv_heads: 2, head_dim: head_dim)
      {blk.w_q, blk.w_k, blk.w_v, blk.w_o}.each { |w| fill_random!(w, Random.new(42)) }

      prompt = SHAInet::SimpleMatrix.new(seq, d_model)
      prng = Random.new(1)
      seq.times { |r| d_model.times { |c| prompt[r, c] = (prng.rand * 2.0 - 1.0) } }

      blk.clear_cache!
      blk.forward_cached(prompt)

      ws = SHAInet::LlamaBlock.attn_ws_floats.to_i64
      ws.should be > 0
      # The bound the feature exists to provide. This is also the regression
      # guard for grow_dev_buf's doubling, which without a cap_limit would
      # allocate up to 2x the budget.
      ws.should be <= budget_floats
      # And it must be far below the unchunked figure, not merely under budget.
      ws.should be < unchunked
    end

    it "bounds the staging buffer by chunk size, not by prompt length" do
      pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

      heads = 8
      head_dim = 8
      d_model = heads * head_dim
      kv_heads = 2
      seq = 2000

      blk = SHAInet::LlamaBlock.new(d_model, heads, 64, num_kv_heads: kv_heads, head_dim: head_dim)
      {blk.w_q, blk.w_k, blk.w_v, blk.w_o}.each { |w| fill_random!(w, Random.new(42)) }

      prompt = SHAInet::SimpleMatrix.new(seq, d_model)
      prng = Random.new(2)
      seq.times { |r| d_model.times { |c| prompt[r, c] = (prng.rand * 2.0 - 1.0) } }

      blk.clear_cache!
      blk.forward_cached(prompt)

      # Staging holds one chunk of K, V and Q, never the whole prompt.
      chunk = blk.attn_chunk_tokens(seq)
      per_chunk = 2 * kv_heads * chunk * head_dim + chunk * d_model
      whole_prompt = 2 * kv_heads * seq * head_dim + seq * d_model

      SHAInet::LlamaBlock.attn_staging_floats.should be >= per_chunk
      SHAInet::LlamaBlock.attn_staging_floats.should be < whole_prompt
    end
  end
end
