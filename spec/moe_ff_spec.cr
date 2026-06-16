require "./spec_helper"

# Build a SwiGLU expert with deterministic, distinct weights (so each expert
# produces a different output and we can verify the combine).
private def fill_expert(ex : SHAInet::SwiGLUFF, d_model : Int32, ff : Int32, seed : Float64)
  g = SHAInet::SimpleMatrix.new(d_model, ff)
  u = SHAInet::SimpleMatrix.new(d_model, ff)
  dn = SHAInet::SimpleMatrix.new(ff, d_model)
  d_model.times { |r| ff.times { |c| g[r, c] = 0.05 * (r + 1) + 0.01 * c + seed } }
  d_model.times { |r| ff.times { |c| u[r, c] = 0.03 * (c + 1) - 0.02 * r + seed } }
  ff.times { |r| d_model.times { |c| dn[r, c] = 0.04 * (r + 1) + 0.02 * c - seed } }
  ex.gate_proj = g
  ex.up_proj = u
  ex.down_proj = dn
end

# Router that makes logit(expert e) == column-sum of router[:, e] when the input
# row is all ones. We put the whole logit in row 0 and zero the rest.
private def router_with_logits(d_model : Int32, logits : Array(Float64)) : SHAInet::SimpleMatrix
  r = SHAInet::SimpleMatrix.new(d_model, logits.size)
  logits.each_with_index { |v, e| r[0, e] = v }
  r
end

describe SHAInet::MoEFF do
  it "selects the top-k experts and combines with renormalized softmax weights" do
    d_model = 4
    ff = 3
    ne = 4
    moe = SHAInet::MoEFF.new(d_model, ff, ne, 2, norm_topk_prob: true)

    # logits: e0=0, e1=2, e2=-5, e3=4  -> top-2 are e3 and e1
    moe.router = router_with_logits(d_model, [0.0, 2.0, -5.0, 4.0])
    moe.experts.each_with_index { |ex, i| fill_expert(ex, d_model, ff, 0.1 * (i + 1)) }

    x = SHAInet::SimpleMatrix.new(1, d_model)
    d_model.times { |c| x[0, c] = 1.0 }

    # Renormalized weights over the two selected experts.
    e3 = Math.exp(4.0); e1 = Math.exp(2.0); denom = e3 + e1
    w3 = e3 / denom; w1 = e1 / denom

    ref3 = moe.experts[3].forward(x)
    ref1 = moe.experts[1].forward(x)

    out = moe.forward(x)
    out.rows.should eq(1)
    out.cols.should eq(d_model)
    d_model.times do |c|
      expected = w3 * ref3[0, c] + w1 * ref1[0, c]
      out[0, c].should be_close(expected, 1e-5)
    end
  end

  it "uses raw (un-renormalized) softmax weights when norm_topk_prob is false" do
    d_model = 4
    ff = 3
    ne = 4
    moe = SHAInet::MoEFF.new(d_model, ff, ne, 2, norm_topk_prob: false)
    logits = [0.0, 2.0, -5.0, 4.0]
    moe.router = router_with_logits(d_model, logits)
    moe.experts.each_with_index { |ex, i| fill_expert(ex, d_model, ff, 0.1 * (i + 1)) }

    x = SHAInet::SimpleMatrix.new(1, d_model)
    d_model.times { |c| x[0, c] = 1.0 }

    z = logits.sum { |l| Math.exp(l) } # full-softmax denominator
    w3 = Math.exp(4.0) / z
    w1 = Math.exp(2.0) / z

    ref3 = moe.experts[3].forward(x)
    ref1 = moe.experts[1].forward(x)

    out = moe.forward(x)
    d_model.times do |c|
      expected = w3 * ref3[0, c] + w1 * ref1[0, c]
      out[0, c].should be_close(expected, 1e-5)
    end
  end

  it "routes each token in a batch independently" do
    d_model = 4
    ff = 3
    ne = 3
    moe = SHAInet::MoEFF.new(d_model, ff, ne, 1, norm_topk_prob: true)
    moe.experts.each_with_index { |ex, i| fill_expert(ex, d_model, ff, 0.2 * (i + 1)) }

    # Router favors expert 0 for [1,0,..] inputs and expert 2 for [0,..,1].
    r = SHAInet::SimpleMatrix.new(d_model, ne)
    r[0, 0] = 10.0 # row-0 feature -> expert 0
    r[d_model - 1, 2] = 10.0 # last feature -> expert 2
    moe.router = r

    x = SHAInet::SimpleMatrix.new(2, d_model)
    x[0, 0] = 1.0          # token 0 -> expert 0
    x[1, d_model - 1] = 1.0 # token 1 -> expert 2

    row0 = SHAInet::SimpleMatrix.new(1, d_model); row0[0, 0] = 1.0
    row1 = SHAInet::SimpleMatrix.new(1, d_model); row1[0, d_model - 1] = 1.0
    ref0 = moe.experts[0].forward(row0) # top_k=1, weight renorm -> 1.0
    ref2 = moe.experts[2].forward(row1)

    out = moe.forward(x)
    out.rows.should eq(2)
    d_model.times do |c|
      out[0, c].should be_close(ref0[0, c], 1e-5)
      out[1, c].should be_close(ref2[0, c], 1e-5)
    end
  end

  it "rejects an invalid top_k" do
    expect_raises(ArgumentError) { SHAInet::MoEFF.new(4, 3, 4, 0) }
    expect_raises(ArgumentError) { SHAInet::MoEFF.new(4, 3, 4, 5) }
  end
end
