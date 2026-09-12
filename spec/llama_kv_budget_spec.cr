require "./spec_helper"

# The device KV cache grows by doubling, which is fine on the first allocation
# (sized to the prompt) but overshoots badly on the first decode token after a
# prefill that exactly filled it: a measured 4096-token prefill allocated 4096
# slots, then token 4097 reallocated to 8192 and half the cache was never used.
#
# kv_max_context replaces that with a single allocation at a stated budget.
private def fill_random!(w, rng : Random)
  m = w.as(SHAInet::SimpleMatrix)
  m.rows.times { |r| m.cols.times { |c| m[r, c] = (rng.rand * 0.2 - 0.1) } }
end

private def build_block(d_model : Int32, heads : Int32, kv_heads : Int32, head_dim : Int32) : SHAInet::LlamaBlock
  blk = SHAInet::LlamaBlock.new(d_model, heads, d_model * 2, num_kv_heads: kv_heads, head_dim: head_dim)
  {blk.w_q, blk.w_k, blk.w_v, blk.w_o}.each { |w| fill_random!(w, Random.new(42)) }
  blk
end

private def random_matrix(rows : Int32, cols : Int32, seed : Int32) : SHAInet::SimpleMatrix
  m = SHAInet::SimpleMatrix.new(rows, cols)
  rng = Random.new(seed)
  rows.times { |r| cols.times { |c| m[r, c] = (rng.rand * 2.0 - 1.0) } }
  m
end

# Bytes the cache should occupy for a given capacity, straight from the layout.
private def expected_bytes(kv_heads : Int32, cap : Int32, head_dim : Int32, fp16 : Bool) : UInt64
  2_u64 * kv_heads.to_u64 * cap.to_u64 * head_dim.to_u64 * (fp16 ? 2_u64 : 4_u64)
end

private def with_env(key : String, value : String?, &)
  prev = ENV[key]?
  if value
    ENV[key] = value
  else
    ENV.delete(key)
  end
  begin
    yield
  ensure
    if prev
      ENV[key] = prev
    else
      ENV.delete(key)
    end
  end
end

describe "LlamaBlock KV context budget" do
  it "allocates exactly the budget and does not grow through it" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    d_model = 64
    head_dim = 8
    heads = 8
    kv_heads = 2
    budget = 600

    blk = build_block(d_model, heads, kv_heads, head_dim)
    blk.kv_max_context = budget
    blk.clear_cache!

    # A 300-token prefill: without a budget this would allocate exactly 300.
    blk.forward_cached(random_matrix(300, d_model, 3))
    after_prefill = blk.kv_cache_bytes
    after_prefill.should eq expected_bytes(kv_heads, budget, head_dim, blk.kv_cache_fp16?)

    # Decoding past the prompt length is exactly where doubling would reallocate
    # to 600; with the budget already in place nothing changes.
    5.times { blk.forward_cached(random_matrix(1, d_model, 4)) }
    blk.kv_cache_bytes.should eq after_prefill
  end

  it "reclaims the overshoot the doubling path leaves behind" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    d_model = 64
    head_dim = 8
    heads = 8
    kv_heads = 2
    prompt = 512 # a prefill that exactly fills a doubling allocation

    # Unbudgeted: prefill allocates 512, then the next token forces 1024.
    grown = build_block(d_model, heads, kv_heads, head_dim)
    grown.clear_cache!
    grown.forward_cached(random_matrix(prompt, d_model, 5))
    at_prefill = grown.kv_cache_bytes
    grown.forward_cached(random_matrix(1, d_model, 6))
    after_one_token = grown.kv_cache_bytes
    after_one_token.should eq at_prefill * 2 # the waste this feature removes

    # Budgeted to what is actually needed: one allocation, no doubling.
    budgeted = build_block(d_model, heads, kv_heads, head_dim)
    budgeted.kv_max_context = prompt + 1
    budgeted.clear_cache!
    budgeted.forward_cached(random_matrix(prompt, d_model, 5))
    budgeted.forward_cached(random_matrix(1, d_model, 6))
    budgeted.kv_cache_bytes.should be < after_one_token
    budgeted.kv_cache_bytes.should eq expected_bytes(kv_heads, prompt + 1, head_dim, budgeted.kv_cache_fp16?)
  end

  it "produces identical output with and without a budget" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    d_model = 64
    head_dim = 8
    heads = 8
    kv_heads = 2
    prompt = random_matrix(48, d_model, 7)
    steps = Array.new(4) { |i| random_matrix(1, d_model, 100 + i) }

    plain = build_block(d_model, heads, kv_heads, head_dim)
    plain.clear_cache!
    plain_outs = [plain.forward_cached(prompt)] + steps.map { |s| plain.forward_cached(s) }

    budgeted = build_block(d_model, heads, kv_heads, head_dim)
    budgeted.kv_max_context = 256
    budgeted.clear_cache!
    budgeted_outs = [budgeted.forward_cached(prompt)] + steps.map { |s| budgeted.forward_cached(s) }

    plain_outs.each_with_index do |want, i|
      got = budgeted_outs[i]
      worst = 0.0
      want.rows.times do |r|
        want.cols.times { |c| d = (want[r, c] - got[r, c]).abs; worst = d if d > worst }
      end
      worst.should be < 1e-6
    end
  end

  it "refuses a context longer than the budget instead of silently truncating" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    blk = build_block(64, 8, 2, 8)
    blk.kv_max_context = 32
    blk.clear_cache!
    expect_raises(ArgumentError, /exceeds the configured KV budget/) do
      blk.forward_cached(random_matrix(64, 64, 9))
    end
  end

  it "validates the budget and refuses to shrink below an existing allocation" do
    blk = build_block(64, 8, 2, 8)
    expect_raises(ArgumentError, /must be positive/) { blk.kv_max_context = 0 }
    expect_raises(ArgumentError, /must be positive/) { blk.kv_max_context = -5 }
    blk.kv_max_context = 4096
    blk.kv_max_context.should eq 4096
    blk.kv_max_context = nil
    blk.kv_max_context.should be_nil
  end

  it "defaults the budget from SHAINET_KV_MAX_CONTEXT" do
    with_env("SHAINET_KV_MAX_CONTEXT", "1234") do
      build_block(64, 8, 2, 8).kv_max_context.should eq 1234
    end
    with_env("SHAINET_KV_MAX_CONTEXT", "not-a-number") do
      build_block(64, 8, 2, 8).kv_max_context.should be_nil
    end
    with_env("SHAINET_KV_MAX_CONTEXT", "0") do
      build_block(64, 8, 2, 8).kv_max_context.should be_nil
    end
    with_env("SHAINET_KV_MAX_CONTEXT", nil) do
      build_block(64, 8, 2, 8).kv_max_context.should be_nil
    end
  end
end
