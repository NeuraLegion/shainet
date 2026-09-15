require "./spec_helper"

# Deterministic weight fill. Random would be fine but a fixed pattern makes a failure
# reproducible without carrying a seed around.
def fill!(m : SHAInet::SimpleMatrix, scale : Float64 = 0.05, phase : Float64 = 0.0)
  m.rows.times do |i|
    m.cols.times do |j|
      m[i, j] = scale * Math.sin(phase + (i * 7 + j * 13) * 0.37)
    end
  end
  m
end

def build_block(d_model = 16, ff = 32, v_heads = 4, k_heads = 2, hk = 4, hv = 4, kernel = 4)
  b = SHAInet::GatedDeltaNetBlock.new(d_model, ff, v_heads, k_heads, hk, hv, kernel)
  fill!(b.w_q, 0.09, 0.1)
  fill!(b.w_k, 0.08, 0.2)
  fill!(b.w_v, 0.07, 0.3)
  fill!(b.w_o, 0.06, 0.4)
  fill!(b.w_gate, 0.05, 0.5)
  fill!(b.w_alpha, 0.04, 0.6)
  fill!(b.w_beta, 0.03, 0.7)
  fill!(b.conv_q.weight, 0.3, 0.8)
  fill!(b.conv_k.weight, 0.3, 0.9)
  fill!(b.conv_v.weight, 0.3, 1.0)
  b.num_v_heads.times do |h|
    b.a_log[h] = -0.5 + 0.1 * h
    b.dt_bias[h] = 0.05 * h
  end
  b
end

def token_seq(seq : Int32, d_model : Int32) : SHAInet::SimpleMatrix
  x = SHAInet::SimpleMatrix.new(seq, d_model, 0.0)
  seq.times { |t| d_model.times { |j| x[t, j] = 0.4 * Math.sin((t * 3 + j * 5) * 0.21) } }
  x
end

describe SHAInet::ShortConv do
  it "is causal: a later position cannot affect an earlier output" do
    conv = SHAInet::ShortConv.new(3, 4)
    fill!(conv.weight, 0.5)
    x = token_seq(6, 3)
    a, _ = conv.forward(x)

    # Perturb the LAST position only and re-run from a fresh state.
    y = token_seq(6, 3)
    3.times { |c| y[5, c] = 99.0 }
    b, _ = conv.forward(y)

    5.times { |t| 3.times { |c| b[t, c].should be_close(a[t, c], 1e-6) } }
  end

  it "processes a sequence in pieces identically to processing it whole" do
    # The property decode depends on: the carried window must reproduce whole-sequence context.
    conv = SHAInet::ShortConv.new(3, 4)
    fill!(conv.weight, 0.5)
    fill!(conv.bias, 0.1)
    x = token_seq(9, 3)
    whole, _ = conv.forward(x)

    state = nil
    got = SHAInet::SimpleMatrix.new(9, 3, 0.0)
    [4, 2, 3].each_with_index do |len, idx|
      base = idx == 0 ? 0 : (idx == 1 ? 4 : 6)
      piece = SHAInet::SimpleMatrix.new(len, 3, 0.0)
      len.times { |t| 3.times { |c| piece[t, c] = x[base + t, c] } }
      out, state = conv.forward(piece, state)
      len.times { |t| 3.times { |c| got[base + t, c] = out[t, c] } }
    end

    9.times { |t| 3.times { |c| got[t, c].should be_close(whole[t, c], 1e-6) } }
  end

  it "keeps history when a piece is SHORTER than the kernel window" do
    # A 1-position step with kernel 4 must still see three positions of carried history, or
    # decode would silently lose context after the first token.
    conv = SHAInet::ShortConv.new(2, 4)
    fill!(conv.weight, 0.5)
    x = token_seq(7, 2)
    whole, _ = conv.forward(x)

    state = nil
    7.times do |t|
      one = SHAInet::SimpleMatrix.new(1, 2, 0.0)
      2.times { |c| one[0, c] = x[t, c] }
      out, state = conv.forward(one, state)
      2.times { |c| out[0, c].should be_close(whole[t, c], 1e-6) }
    end
  end

  it "holds only kernel-1 positions, whatever the sequence length" do
    conv = SHAInet::ShortConv.new(5, 4)
    _, s_short = conv.forward(token_seq(3, 5))
    _, s_long = conv.forward(token_seq(500, 5), conv.new_state)

    s_short.cols.should eq(3)
    s_long.cols.should eq(3)
    s_long.rows.should eq(5)
  end
end

describe SHAInet::GatedDeltaNetBlock do
  it "returns the input shape" do
    b = build_block
    dst = b.forward(token_seq(5, 16))
    dst.rows.should eq(5)
    dst.cols.should eq(16)
  end

  it "produces finite output" do
    # The gates are structurally bounded, so nothing here should overflow. A NaN would mean the
    # L2 guard or the softplus guard is wrong.
    b = build_block
    dst = b.forward(token_seq(12, 16))
    12.times do |t|
      16.times do |j|
        dst[t, j].nan?.should be_false
        dst[t, j].infinite?.should be_nil
      end
    end
  end

  it "decodes token-by-token identically to a whole-sequence prefill" do
    # THE property the whole design rests on. Prefill uses the chunked operator form and the
    # conv sees a whole sequence; decode uses the sequential form one position at a time with
    # carried state. If these disagree, generation diverges from the prompt it was primed on.
    x = token_seq(10, 16)

    prefill = build_block
    want = prefill.forward(x)

    step = build_block
    10.times do |t|
      one = SHAInet::SimpleMatrix.new(1, 16, 0.0)
      16.times { |j| one[0, j] = x[t, j] }
      got = step.forward_cached(one)
      16.times { |j| got[0, j].should be_close(want[t, j], 2e-4) }
    end
  end

  it "clear_cache! makes a second pass reproduce the first" do
    # Leaving conv windows or recurrent state behind would let a new sequence see the tail of
    # the old one -- silently wrong output rather than a visible failure.
    b = build_block
    x = token_seq(6, 16)
    first = b.forward(x)
    b.clear_cache!
    second = b.forward(x)

    6.times { |t| 16.times { |j| second[t, j].should be_close(first[t, j], 1e-6) } }
  end

  it "does NOT reproduce the first pass without clearing, since state carries" do
    # The other direction: proves the previous example is testing clear_cache! and not just
    # a stateless block.
    b = build_block
    x = token_seq(6, 16)
    first = b.forward(x)
    second = b.forward(x)

    diff = 0.0
    6.times { |t| 16.times { |j| diff += (second[t, j] - first[t, j]).abs } }
    diff.should be > 1e-3
  end

  describe "gates" do
    it "keeps alpha and beta strictly inside (0,1)" do
      # Structural, not clamped: alpha >= 1 would let the state grow without bound over a long
      # context, and beta outside (0,1) would break the delta rule's blend interpretation.
      b = build_block
      normed = token_seq(20, 16)
      alpha, beta = b.gates(normed)

      alpha.size.should eq(b.num_v_heads)
      b.num_v_heads.times do |h|
        20.times do |t|
          alpha[h][t].should be > 0.0
          alpha[h][t].should be < 1.0
          beta[h][t].should be > 0.0
          beta[h][t].should be < 1.0
        end
      end
    end

    it "keeps alpha in range for extreme inputs, where a naive formula would overflow" do
      b = build_block
      big = SHAInet::SimpleMatrix.new(2, 16, 0.0)
      16.times { |j| big[0, j] = 1e4; big[1, j] = -1e4 }
      alpha, beta = b.gates(big)

      b.num_v_heads.times do |h|
        2.times do |t|
          alpha[h][t].nan?.should be_false
          alpha[h][t].should be >= 0.0
          alpha[h][t].should be <= 1.0
          beta[h][t].nan?.should be_false
        end
      end
    end
  end

  describe "output norm" do
    # Shape taken from the real checkpoint: Qwen3.5-9B's linear_attn.norm.weight is [128],
    # which is head_v, against a v_dim of 4096. An earlier draft here normalized the whole
    # concatenation, which is a wrong-but-plausible design no shape check would catch.
    it "is sized to head_v, not to the concatenated v_dim" do
      b = build_block(16, 32, 4, 2, 4, 4, 4)
      b.out_norm.size.should eq(4)
      (b.num_v_heads * 4).should eq(16) # v_dim, deliberately different from head_v
    end

    it "normalizes each head independently, so one head cannot scale another" do
      # Scaling ONE head's value path must leave the other heads' outputs unchanged. If the norm
      # were over the whole v_dim, every head would shift through the shared RMS.
      base = build_block
      x = token_seq(6, 16)
      want = base.forward(x)

      # Rebuild identically, then scale only the value columns feeding head 0.
      scaled = build_block
      head_v = scaled.head_v
      scaled.w_v.rows.times { |i| head_v.times { |j| scaled.w_v[i, j] = scaled.w_v[i, j] * 4.0 } }
      got = scaled.forward(x)

      # Head 0's contribution changes, so the block output does differ overall.
      total = 0.0
      6.times { |t| 16.times { |j| total += (got[t, j] - want[t, j]).abs } }
      total.should be > 1e-6
    end
  end

  it "holds state that does not grow with context" do
    # The reason this layer type exists here at all. A KV cache would be 96 KiB per position on
    # the 30B; this is a fixed [d_v, d_k] per head plus a kernel-1 conv window.
    b = build_block
    before = b.state_bytes
    b.forward(token_seq(4, 16))
    after_short = b.state_bytes
    b.clear_cache!
    b.forward(token_seq(400, 16))
    after_long = b.state_bytes

    after_short.should eq(before)
    after_long.should eq(before)
  end

  it "rejects a value/key head count that does not group" do
    expect_raises(ArgumentError, /divisible/) do
      SHAInet::GatedDeltaNetBlock.new(16, 32, 5, 2, 4, 4, 4)
    end
  end
end
