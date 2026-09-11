require "./spec_helper"

# Performance guards for the Phase 1 changes. These assert that reclaiming VRAM
# did not cost throughput, which the correctness specs say nothing about.
#
# Timing on a shared GPU is noisy, so every measurement here takes the BEST of N
# runs after a warm-up (min, not mean: the floor is the signal, spikes are the
# machine) and asserts a generous ratio rather than an absolute duration. They
# are tagged "perf" so a noisy CI can exclude them:
#
#   crystal spec --tag ~perf      # skip
#   crystal spec --tag perf       # only these
private def fill_random!(w, rng : Random)
  m = w.as(SHAInet::SimpleMatrix)
  m.rows.times { |r| m.cols.times { |c| m[r, c] = (rng.rand * 0.2 - 0.1) } }
end

private def build_block(d_model : Int32, heads : Int32, kv_heads : Int32) : SHAInet::LlamaBlock
  blk = SHAInet::LlamaBlock.new(d_model, heads, d_model * 2, num_kv_heads: kv_heads)
  {blk.w_q, blk.w_k, blk.w_v, blk.w_o}.each { |w| fill_random!(w, Random.new(42)) }
  ffn = blk.ffn.as(SHAInet::SwiGLUFF)
  fill_random!(ffn.gate_proj, Random.new(7))
  fill_random!(ffn.up_proj, Random.new(8))
  fill_random!(ffn.down_proj, Random.new(9))
  blk
end

private def random_matrix(rows : Int32, cols : Int32, seed : Int32) : SHAInet::SimpleMatrix
  m = SHAInet::SimpleMatrix.new(rows, cols)
  rng = Random.new(seed)
  rows.times { |r| cols.times { |c| m[r, c] = (rng.rand * 2.0 - 1.0) } }
  m
end

# Best-of-n wall time for a block of work, after one warm-up iteration.
private def best_of(n : Int32, &block : ->) : Time::Span
  block.call # warm up: first call pays kernel load, buffer growth, JIT of nothing
  best = Time::Span::MAX
  n.times do
    t0 = Time.instant
    block.call
    dt = Time.instant - t0
    best = dt if dt < best
  end
  best
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

describe "Phase 1 performance guards" do
  it "chunked prefill costs little against a single-chunk prefill", tags: "perf" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    d_model = 256
    heads = 8
    seq = 512
    prompt = random_matrix(seq, d_model, 3)

    single = with_env("SHAINET_ATTN_CHUNK", seq.to_s) do
      blk = build_block(d_model, heads, 4)
      best_of(3) do
        blk.clear_cache!
        blk.forward_cached(prompt)
      end
    end

    chunked = with_env("SHAINET_ATTN_CHUNK", "64") do
      blk = build_block(d_model, heads, 4)
      best_of(3) do
        blk.clear_cache!
        blk.forward_cached(prompt)
      end
    end

    ratio = chunked.total_milliseconds / single.total_milliseconds
    # 8 chunks means 8x the kernel launches and H2D copies for the same math, so
    # some overhead is expected and acceptable. A blow-up past 3x would mean the
    # chunk loop is doing real redundant work, not just paying launch overhead.
    ratio.should be < 3.0
  end

  it "keeps decode throughput intact when prefill was chunked", tags: "perf" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    d_model = 256
    heads = 8
    prompt = random_matrix(256, d_model, 4)
    token = random_matrix(1, d_model, 5)

    # Decode is a single token, so chunking must be a complete no-op on this path.
    single = with_env("SHAINET_ATTN_CHUNK", "1024") do
      blk = build_block(d_model, heads, 4)
      blk.clear_cache!
      blk.forward_cached(prompt)
      best_of(5) { blk.forward_cached(token) }
    end

    chunked = with_env("SHAINET_ATTN_CHUNK", "32") do
      blk = build_block(d_model, heads, 4)
      blk.clear_cache!
      blk.forward_cached(prompt)
      best_of(5) { blk.forward_cached(token) }
    end

    ratio = chunked.total_milliseconds / single.total_milliseconds
    ratio.should be < 1.5
  end

  it "does not make attention slower by storing the KV cache in fp16", tags: "perf" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    d_model = 256
    heads = 8
    prompt = random_matrix(512, d_model, 6)
    token = random_matrix(1, d_model, 7)

    fp32 = with_env("SHAINET_KV_FP16", "0") do
      blk = build_block(d_model, heads, 4)
      blk.clear_cache!
      blk.forward_cached(prompt)
      best_of(5) { blk.forward_cached(token) }
    end

    fp16 = with_env("SHAINET_KV_FP16", "1") do
      blk = build_block(d_model, heads, 4)
      blk.clear_cache!
      blk.forward_cached(prompt)
      best_of(5) { blk.forward_cached(token) }
    end

    ratio = fp16.total_milliseconds / fp32.total_milliseconds
    # Attention over the cache is bandwidth-bound, so halving the bytes read
    # should be neutral-to-faster. The extra __half2float per element must not
    # turn into a regression.
    ratio.should be < 1.25
  end

  it "does not slow token embedding by keeping the table in host RAM", tags: "perf" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    vocab = 32_000
    dim = 512
    ids = [17, 4096, 31_999, 3]

    device_layer = SHAInet::EmbeddingLayer.new(vocab, dim)
    on_device = best_of(20) { device_layer.embed(ids) }

    host_layer = SHAInet::EmbeddingLayer.new(vocab, dim)
    host_layer.to_host!
    on_host = best_of(20) { host_layer.embed(ids) }

    ratio = on_host.total_milliseconds / on_device.total_milliseconds
    # A decode step gathers a handful of rows (tens of KB). The host path adds a
    # memcpy of exactly that slice, which must stay negligible; if this fails the
    # gather is copying far more than the requested rows.
    ratio.should be < 3.0
  end

  it "keeps the host embedding gather proportional to the batch, not the vocab", tags: "perf" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    dim = 512
    small_vocab = SHAInet::EmbeddingLayer.new(8_000, dim)
    small_vocab.to_host!
    large_vocab = SHAInet::EmbeddingLayer.new(64_000, dim)
    large_vocab.to_host!

    ids = [1, 2, 3, 4]
    small = best_of(20) { small_vocab.embed(ids) }
    large = best_of(20) { large_vocab.embed(ids) }

    ratio = large.total_milliseconds / small.total_milliseconds
    # An 8x bigger table gathering the same 4 rows must cost the same. This is
    # the regression guard against ever reintroducing a whole-table copy.
    ratio.should be < 2.0
  end
end
