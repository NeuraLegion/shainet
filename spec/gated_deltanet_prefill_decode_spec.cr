require "./spec_helper"

# Generation prefills a GatedDeltaNetBlock with the CHUNKED operator form and then decodes with
# the SEQUENTIAL one, carrying the recurrent state and the conv window across. If those two forms
# disagree at block level the model is inconsistent with itself mid-sequence, and over 24 linear
# layers the error compounds until the hidden state washes out.
#
# spec/gated_deltanet_spec.cr already pins chunked == recurrent for the bare operator. This pins
# it for the BLOCK, which adds the short conv, the SiLU, the L2 normalization, the per-head output
# norm and the gate -- each of which carries its own state or ordering.

def gpd_fill!(m : SHAInet::SimpleMatrix, seed : Int32)
  r = Random.new(seed)
  m.rows.times { |i| m.cols.times { |j| m[i, j] = (r.next_float - 0.5) * 0.4 } }
  m
end

def gpd_block(d = 32, ff = 64)
  b = SHAInet::GatedDeltaNetBlock.new(d, ff, num_v_heads: 4, num_k_heads: 2, head_k: 8, head_v: 8, conv_kernel: 4)
  gpd_fill!(b.w_q.as(SHAInet::SimpleMatrix), 1)
  gpd_fill!(b.w_k.as(SHAInet::SimpleMatrix), 2)
  gpd_fill!(b.w_v.as(SHAInet::SimpleMatrix), 3)
  gpd_fill!(b.w_o.as(SHAInet::SimpleMatrix), 4)
  gpd_fill!(b.w_gate.as(SHAInet::SimpleMatrix), 5)
  gpd_fill!(b.w_alpha.as(SHAInet::SimpleMatrix), 6)
  gpd_fill!(b.w_beta.as(SHAInet::SimpleMatrix), 7)
  gpd_fill!(b.conv_q.weight, 8)
  gpd_fill!(b.conv_k.weight, 9)
  gpd_fill!(b.conv_v.weight, 10)
  ffn = b.ffn
  gpd_fill!(ffn.gate_proj.as(SHAInet::SimpleMatrix), 11)
  gpd_fill!(ffn.up_proj.as(SHAInet::SimpleMatrix), 12)
  gpd_fill!(ffn.down_proj.as(SHAInet::SimpleMatrix), 13)
  b.num_v_heads.times { |h| b.a_log[h] = -0.4 + h * 0.15; b.dt_bias[h] = 0.05 }
  b
end

# SimpleMatrix is Float32 and the two forms sum in different orders, so this bounds the
# disagreement rather than demanding equality. It is tight enough that a wrong operator form
# fails: the real 9B mismatch was orders of magnitude larger than this.
GPD_TOL = 2e-4

describe "GatedDeltaNetBlock prefill / decode equivalence" do
  it "matches a whole-sequence forward against token-at-a-time forward_cached" do
    seq = 7
    x = gpd_fill!(SHAInet::SimpleMatrix.new(seq, 32), 30)
    ref = gpd_block.forward(x)

    b = gpd_block
    b.clear_cache!
    got = SHAInet::SimpleMatrix.new(seq, 32)
    seq.times do |t|
      row = SHAInet::SimpleMatrix.new(1, 32)
      32.times { |j| row[0, j] = x[t, j] }
      r = b.forward_cached(row)
      32.times { |j| got[t, j] = r[0, j] }
    end

    worst = 0.0
    seq.times { |t| 32.times { |j| worst = Math.max(worst, (got[t, j].to_f64 - ref[t, j].to_f64).abs) } }
    worst.should be < GPD_TOL
  end

  it "matches a chunked prefill followed by a single decode step, which is what generation does" do
    seq = 6
    x = gpd_fill!(SHAInet::SimpleMatrix.new(seq, 32), 31)
    ref = gpd_block.forward(x)

    b = gpd_block
    b.clear_cache!
    head = SHAInet::SimpleMatrix.new(seq - 1, 32)
    (seq - 1).times { |t| 32.times { |j| head[t, j] = x[t, j] } }
    b.forward(head)
    last = SHAInet::SimpleMatrix.new(1, 32)
    32.times { |j| last[0, j] = x[seq - 1, j] }
    got = b.forward_cached(last)

    worst = 0.0
    32.times { |j| worst = Math.max(worst, (got[0, j].to_f64 - ref[seq - 1, j].to_f64).abs) }
    worst.should be < GPD_TOL
  end

  it "diverges when the state is NOT carried, so the checks above are not vacuous" do
    # Clearing between steps must break the equivalence. Without this, a forward_cached that
    # ignored the carried state entirely could still satisfy both examples if the state happened
    # to contribute little.
    seq = 6
    x = gpd_fill!(SHAInet::SimpleMatrix.new(seq, 32), 31)
    ref = gpd_block.forward(x)

    b = gpd_block
    got = SHAInet::SimpleMatrix.new(seq, 32)
    seq.times do |t|
      b.clear_cache!
      row = SHAInet::SimpleMatrix.new(1, 32)
      32.times { |j| row[0, j] = x[t, j] }
      r = b.forward_cached(row)
      32.times { |j| got[t, j] = r[0, j] }
    end

    worst = 0.0
    seq.times { |t| 32.times { |j| worst = Math.max(worst, (got[t, j].to_f64 - ref[t, j].to_f64).abs) } }
    worst.should be > GPD_TOL
  end
end
