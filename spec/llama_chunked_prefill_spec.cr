require "./spec_helper"

# Prefill is attended in bounded chunks so the attention workspace stays
# O(num_heads * chunk * total_len) instead of O(num_heads * seq^2). These specs
# pin the invariant that chunking is transparent: forcing a tiny chunk (so the
# loop runs many times) must reproduce both the unchunked GPU result and the CPU
# reference, including the RoPE position re-basing inside each chunk.
private def fill_random!(w, rng : Random)
  m = w.as(SHAInet::SimpleMatrix)
  m.rows.times { |r| m.cols.times { |c| m[r, c] = (rng.rand * 0.2 - 0.1) } }
end

private def build_block(rng : Random, d_model : Int32) : SHAInet::LlamaBlock
  blk = SHAInet::LlamaBlock.new(d_model, 8, 128, num_kv_heads: 4)
  {blk.w_q, blk.w_k, blk.w_v, blk.w_o}.each { |w| fill_random!(w, rng) }
  ffn = blk.ffn.as(SHAInet::SwiGLUFF)
  fill_random!(ffn.gate_proj, rng)
  fill_random!(ffn.up_proj, rng)
  fill_random!(ffn.down_proj, rng)
  blk
end

private def max_abs_diff(a : SHAInet::SimpleMatrix, b : SHAInet::SimpleMatrix) : Float64
  worst = 0.0
  a.rows.times do |r|
    a.cols.times do |c|
      d = (a[r, c] - b[r, c]).abs
      worst = d if d > worst
    end
  end
  worst
end

private def with_attn_chunk(value : String?, &)
  prev = ENV["SHAINET_ATTN_CHUNK"]?
  if value
    ENV["SHAINET_ATTN_CHUNK"] = value
  else
    ENV.delete("SHAINET_ATTN_CHUNK")
  end
  begin
    yield
  ensure
    if prev
      ENV["SHAINET_ATTN_CHUNK"] = prev
    else
      ENV.delete("SHAINET_ATTN_CHUNK")
    end
  end
end

describe "LlamaBlock chunked prefill" do
  it "matches the unchunked GPU result when the chunk loop runs many times" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    d_model = 64
    seq = 37 # deliberately not a multiple of the forced chunk size
    prompt = SHAInet::SimpleMatrix.new(seq, d_model)
    prng = Random.new(11)
    seq.times { |r| d_model.times { |c| prompt[r, c] = (prng.rand * 2.0 - 1.0) } }

    # One chunk covering the whole prompt.
    blk = build_block(Random.new(42), d_model)
    single = with_attn_chunk(seq.to_s) do
      blk.clear_cache!
      blk.forward_cached(prompt)
    end

    # Chunk of 5 => 8 iterations with a ragged tail of 2.
    blk2 = build_block(Random.new(42), d_model)
    chunked = with_attn_chunk("5") do
      blk2.clear_cache!
      blk2.forward_cached(prompt)
    end

    max_abs_diff(single, chunked).should be < 1e-4
  end

  it "matches the CPU reference under a chunk size of 1" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    d_model = 64
    seq = 12
    prompt = SHAInet::SimpleMatrix.new(seq, d_model)
    prng = Random.new(5)
    seq.times { |r| d_model.times { |c| prompt[r, c] = (prng.rand * 2.0 - 1.0) } }

    blk = build_block(Random.new(42), d_model)
    blk.force_cpu_attention = true
    blk.clear_cache!
    cpu = blk.forward_cached(prompt)

    blk2 = build_block(Random.new(42), d_model)
    gpu = with_attn_chunk("1") do
      blk2.clear_cache!
      blk2.forward_cached(prompt)
    end

    max_abs_diff(cpu, gpu).should be < 1e-3
  end

  it "keeps decode correct after a chunked prefill" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    d_model = 64
    seq = 20
    prng = Random.new(3)
    prompt = SHAInet::SimpleMatrix.new(seq, d_model)
    seq.times { |r| d_model.times { |c| prompt[r, c] = (prng.rand * 2.0 - 1.0) } }
    steps = Array.new(4) do
      tok = SHAInet::SimpleMatrix.new(1, d_model)
      d_model.times { |c| tok[0, c] = (prng.rand * 2.0 - 1.0) }
      tok
    end

    blk = build_block(Random.new(42), d_model)
    blk.force_cpu_attention = true
    blk.clear_cache!
    blk.forward_cached(prompt)
    cpu_steps = steps.map { |t| blk.forward_cached(t) }

    blk2 = build_block(Random.new(42), d_model)
    gpu_steps = with_attn_chunk("3") do
      blk2.clear_cache!
      blk2.forward_cached(prompt)
      steps.map { |t| blk2.forward_cached(t) }
    end

    cpu_steps.each_with_index do |cpu, i|
      max_abs_diff(cpu, gpu_steps[i]).should be < 1e-3
    end
  end
end
