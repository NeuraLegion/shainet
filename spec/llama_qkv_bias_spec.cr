require "./spec_helper"

# Verifies Qwen2-style Q/K/V projection biases flow through the cached
# (Network#run) attention path on CPU, and that the LLaMA default (no bias)
# is unaffected. Forces the CPU attention path so the test is deterministic
# and independent of CUDA/VRAM.
describe SHAInet::LlamaBlock do
  it "treats nil bias and zero bias identically (LLaMA path unchanged)" do
    d_model = 8
    block_a = SHAInet::LlamaBlock.new(d_model, num_heads: 2, ff_hidden: 16, num_kv_heads: 2)
    block_b = SHAInet::LlamaBlock.new(d_model, num_heads: 2, ff_hidden: 16, num_kv_heads: 2)
    block_a.force_cpu_attention = true
    block_b.force_cpu_attention = true

    # Identical random-ish weights for both blocks.
    seed = SHAInet::SimpleMatrix.new(d_model, d_model)
    d_model.times { |i| d_model.times { |j| seed[i, j] = ((i * 7 + j * 3) % 5 - 2) * 0.1 } }
    [block_a, block_b].each do |bl|
      bl.w_q = seed.clone
      bl.w_k = seed.clone
      bl.w_v = seed.clone
      bl.w_o = seed.clone
      d_model.times { |j| bl.norm1.gamma.not_nil![0, j] = 1.0; bl.norm2.gamma.not_nil![0, j] = 1.0 }
    end

    # b only differs by being nil vs all-zeros — must produce identical output.
    block_b.b_q = Array(Float32).new(d_model, 0.0_f32)
    block_b.b_k = Array(Float32).new(d_model, 0.0_f32)
    block_b.b_v = Array(Float32).new(d_model, 0.0_f32)

    x = SHAInet::SimpleMatrix.new(1, d_model)
    d_model.times { |j| x[0, j] = (j - 4) * 0.05 }

    out_a = block_a.forward_cached(x.clone)
    out_b = block_b.forward_cached(x.clone)

    d_model.times do |j|
      out_b[0, j].should be_close(out_a[0, j], 1e-6)
    end
  end

  it "shifts the output when a non-zero QKV bias is set" do
    d_model = 8
    base = SHAInet::LlamaBlock.new(d_model, num_heads: 2, ff_hidden: 16, num_kv_heads: 2)
    biased = SHAInet::LlamaBlock.new(d_model, num_heads: 2, ff_hidden: 16, num_kv_heads: 2)
    base.force_cpu_attention = true
    biased.force_cpu_attention = true

    seed = SHAInet::SimpleMatrix.new(d_model, d_model)
    d_model.times { |i| d_model.times { |j| seed[i, j] = ((i + j) % 3 - 1) * 0.1 } }
    [base, biased].each do |bl|
      bl.w_q = seed.clone
      bl.w_k = seed.clone
      bl.w_v = seed.clone
      bl.w_o = seed.clone
      d_model.times { |j| bl.norm1.gamma.not_nil![0, j] = 1.0; bl.norm2.gamma.not_nil![0, j] = 1.0 }
    end

    biased.b_q = Array(Float32).new(d_model) { |i| (i + 1) * 0.3_f32 }
    biased.b_k = Array(Float32).new(d_model) { |i| (i + 1) * 0.2_f32 }
    biased.b_v = Array(Float32).new(d_model) { |i| (i + 1) * 0.5_f32 }

    x = SHAInet::SimpleMatrix.new(1, d_model)
    d_model.times { |j| x[0, j] = (j - 4) * 0.05 }

    out_base = base.forward_cached(x.clone)
    out_biased = biased.forward_cached(x.clone)

    # The V bias in particular must move the attention output, hence the block output.
    diff = (0...d_model).sum { |j| (out_biased[0, j] - out_base[0, j]).abs }
    diff.should be > 0.01
  end
end
