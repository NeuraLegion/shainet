require "./spec_helper"

# Fill a matrix with something deterministic and non-degenerate. Constant weights would let a
# wrong slice or a dropped gate still produce the right numbers.
def qhs_fill!(m : SHAInet::SimpleMatrix, seed : Int32)
  r = Random.new(seed)
  m.rows.times { |i| m.cols.times { |j| m[i, j] = (r.next_float - 0.5) * 0.4 } }
  m
end

def qhs_block(d = 32, ff = 64)
  b = SHAInet::GatedDeltaNetBlock.new(d, ff, num_v_heads: 4, num_k_heads: 2, head_k: 8, head_v: 8, conv_kernel: 4)
  qhs_fill!(b.w_q.as(SHAInet::SimpleMatrix), 1)
  qhs_fill!(b.w_k.as(SHAInet::SimpleMatrix), 2)
  qhs_fill!(b.w_v.as(SHAInet::SimpleMatrix), 3)
  qhs_fill!(b.w_o.as(SHAInet::SimpleMatrix), 4)
  qhs_fill!(b.w_gate.as(SHAInet::SimpleMatrix), 5)
  qhs_fill!(b.w_alpha.as(SHAInet::SimpleMatrix), 6)
  qhs_fill!(b.w_beta.as(SHAInet::SimpleMatrix), 7)
  qhs_fill!(b.conv_q.weight, 8)
  qhs_fill!(b.conv_k.weight, 9)
  qhs_fill!(b.conv_v.weight, 10)
  b.num_v_heads.times { |h| b.a_log[h] = -0.5 + h * 0.1; b.dt_bias[h] = 0.1 }
  b
end

describe "quantized hybrid stack" do
  it "reports a fresh block as not quantized" do
    # The other direction of the guard below: a `quantized?` that returned true unconditionally
    # would pass every assertion that matters.
    qhs_block.quantized?.should be_false
  end

  it "quantizes the five large projections and leaves the gate parameters in host fp32" do
    pending! "requires CUDA" unless SHAInet::CUDA.fully_available?
    b = qhs_block
    b.to_gpu!(quantize: true, bits: 4)
    b.quantized?.should be_true
    # w_alpha and w_beta move to the device as fp32 rather than being quantized: they are
    # [d_model, num_v_heads], 0.5 MB against 100 MB for the rest, and alpha feeds
    # exp(-exp(a_log) * softplus(...)) where a Q4 rounding would move the decay itself.
    b.w_alpha.should_not be_a(SHAInet::QuantizedWeight)
    b.w_beta.should_not be_a(SHAInet::QuantizedWeight)
  end

  it "keeps the mixer's output close to fp32 after 4-bit quantization" do
    pending! "requires CUDA" unless SHAInet::CUDA.fully_available?
    x = qhs_fill!(SHAInet::SimpleMatrix.new(6, 32), 11)

    ref = qhs_block.forward(x)
    q = qhs_block
    q.to_gpu!(quantize: true, bits: 4)
    got = q.forward(x)

    got.rows.should eq(ref.rows)
    got.cols.should eq(ref.cols)
    # Q4 is a lossy weight format, so this bounds the error rather than asserting equality. The
    # bound is RELATIVE to the reference's own scale: an absolute tolerance would pass trivially
    # on a block whose output happened to be small.
    scale = 0.0
    ref.rows.times { |i| ref.cols.times { |j| scale = Math.max(scale, ref[i, j].to_f64.abs) } }
    scale.should be > 1e-4
    worst = 0.0
    ref.rows.times { |i| ref.cols.times { |j| worst = Math.max(worst, (got[i, j].to_f64 - ref[i, j].to_f64).abs) } }
    (worst / scale).should be < 0.25
  end

  it "quantizes a hybrid stack's linear-attention layers through Network#quantize!" do
    pending! "requires CUDA" unless SHAInet::CUDA.fully_available?
    net = SHAInet::Network.new
    net.add_layer(:input, 1)
    net.add_layer(:embedding, 32, vocab_size: 40)
    net.add_layer("gated_deltanet", 32, num_heads: 4, ff_hidden: 64, num_kv_heads: 2, head_dim: 8)
    net.add_layer(:llama, 32, num_heads: 4, ff_hidden: 64, num_kv_heads: 2, head_dim: 8)
    net.add_layer("gated_deltanet", 32, num_heads: 4, ff_hidden: 64, num_kv_heads: 2, head_dim: 8)
    net.add_layer(:output, 40, activation_function: SHAInet.identity)
    net.fully_connect

    gdn = net.hidden_layers.select(SHAInet::GatedDeltaNetBlock)
    gdn.size.should eq(2)
    gdn.count(&.quantized?).should eq(0)

    net.quantize!(4)

    # The point of the change: gated_deltanet blocks are not in @transformer_layers, so a
    # quantize! that walked only that array left 24 of Qwen3.5-9B's 32 layers in fp32.
    gdn.count(&.quantized?).should eq(2)
  end
end
