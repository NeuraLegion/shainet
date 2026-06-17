require "./spec_helper"

# Compares the GPU KV-cache attention path (attention_heads_gpu) against the
# CPU reference (attention_heads_cpu) on identical weights and inputs, through
# both prefill (M>1) and incremental decode (M=1), including GQA head mapping.
private def fill_random!(w, rng : Random)
  m = w.as(SHAInet::SimpleMatrix)
  m.rows.times { |r| m.cols.times { |c| m[r, c] = (rng.rand * 0.2 - 0.1) } }
end

private def run_cached(blk : SHAInet::LlamaBlock, inputs : Array(SHAInet::SimpleMatrix)) : Array(SHAInet::SimpleMatrix)
  blk.clear_cache!
  inputs.map { |x| blk.forward_cached(x) }
end

describe "LlamaBlock GPU attention" do
  it "matches the CPU attention path for prefill and decode (GQA)" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    rng = Random.new(42)
    d_model = 64
    blk = SHAInet::LlamaBlock.new(d_model, 8, 128, num_kv_heads: 4)
    {blk.w_q, blk.w_k, blk.w_v, blk.w_o}.each { |w| fill_random!(w, rng) }
    ffn = blk.ffn.as(SHAInet::SwiGLUFF)
    fill_random!(ffn.gate_proj, rng)
    fill_random!(ffn.up_proj, rng)
    fill_random!(ffn.down_proj, rng)

    # Prefill of 6 tokens followed by 3 single-token decode steps.
    inputs = [] of SHAInet::SimpleMatrix
    prefill = SHAInet::SimpleMatrix.new(6, d_model)
    6.times { |r| d_model.times { |c| prefill[r, c] = (rng.rand * 2.0 - 1.0) } }
    inputs << prefill
    3.times do
      tok = SHAInet::SimpleMatrix.new(1, d_model)
      d_model.times { |c| tok[0, c] = (rng.rand * 2.0 - 1.0) }
      inputs << tok
    end

    blk.force_cpu_attention = true
    cpu_outs = run_cached(blk, inputs)
    blk.force_cpu_attention = false
    gpu_outs = run_cached(blk, inputs)

    cpu_outs.each_with_index do |cpu, step|
      gpu = gpu_outs[step]
      max_diff = 0.0
      dot = 0.0
      na = 0.0
      nb = 0.0
      cpu.rows.times do |r|
        cpu.cols.times do |c|
          a = cpu[r, c]
          b = gpu[r, c]
          diff = (a - b).abs
          max_diff = diff if diff > max_diff
          dot += a * b
          na += a * a
          nb += b * b
        end
      end
      cos = dot / (Math.sqrt(na) * Math.sqrt(nb))
      max_diff.should be < 1e-3
      cos.should be > 0.9999
    end
  end

  it "stays correct across a device cache growth and clear_cache!" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    rng = Random.new(7)
    d_model = 32
    blk = SHAInet::LlamaBlock.new(d_model, 4, 64, num_kv_heads: 2)
    {blk.w_q, blk.w_k, blk.w_v, blk.w_o}.each { |w| fill_random!(w, rng) }

    # 300 cached positions forces at least one capacity grow (initial 256).
    inputs = [] of SHAInet::SimpleMatrix
    prefill = SHAInet::SimpleMatrix.new(290, d_model)
    290.times { |r| d_model.times { |c| prefill[r, c] = (rng.rand * 2.0 - 1.0) } }
    inputs << prefill
    10.times do
      tok = SHAInet::SimpleMatrix.new(1, d_model)
      d_model.times { |c| tok[0, c] = (rng.rand * 2.0 - 1.0) }
      inputs << tok
    end

    blk.force_cpu_attention = true
    cpu_outs = run_cached(blk, inputs)
    blk.force_cpu_attention = false
    gpu_outs = run_cached(blk, inputs)
    # Re-run after clear to confirm stale device cache positions never leak.
    gpu_outs2 = run_cached(blk, inputs)

    {gpu_outs, gpu_outs2}.each do |outs|
      cpu_outs.each_with_index do |cpu, step|
        gpu = outs[step]
        max_diff = 0.0
        cpu.rows.times do |r|
          cpu.cols.times { |c| d = (cpu[r, c] - gpu[r, c]).abs; max_diff = d if d > max_diff }
        end
        max_diff.should be < 1e-3
      end
    end
  end
end
