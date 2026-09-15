require "./spec_helper"

# There is no PyTorch on this machine, so these examples cannot diff against a reference
# implementation. Instead each one pins a PROPERTY that distinguishes the gated delta rule
# from something it could be mistaken for -- plain linear attention, pure Mamba2 decay, pure
# DeltaNet. Together they constrain the recurrence tightly enough that a transcription error
# in Eq. 10 fails at least one of them.
#
# This matters more than usual: this operator is the reference that a future GPU kernel and
# chunked prefill form get checked against, so an error here would be inherited silently by
# everything built on it.
def one_hot(seq : Int32, dim : Int32, rows : Array(Int32)) : SHAInet::SimpleMatrix
  m = SHAInet::SimpleMatrix.new(seq, dim, 0.0)
  rows.each_with_index { |col, t| m[t, col] = 1.0 }
  m
end

describe SHAInet::GatedDeltaNet do
  describe "associative recall" do
    it "returns the value written for a key when queried with that key" do
      # The defining behaviour: write (k=e0, v) at t=0 with full write strength and no decay,
      # then query with the same key at t=1 and expect v back.
      k = one_hot(2, 4, [0, 0])
      q = one_hot(2, 4, [0, 0])
      v = SHAInet::SimpleMatrix.new(2, 3, 0.0)
      v[0, 0] = 0.5; v[0, 1] = -0.25; v[0, 2] = 2.0
      # Second token writes nothing (beta 0) so it only reads.
      dst, _ = SHAInet::GatedDeltaNet.recurrent(q, k, v, [1.0, 1.0], [1.0, 0.0])

      dst[1, 0].should be_close(0.5, 1e-9)
      dst[1, 1].should be_close(-0.25, 1e-9)
      dst[1, 2].should be_close(2.0, 1e-9)
    end

    it "keeps two orthogonal keys independently retrievable" do
      # A fixed-size state can hold at most d_k orthogonal associations. Two must be exact.
      k = one_hot(4, 4, [0, 1, 0, 1])
      q = one_hot(4, 4, [0, 1, 0, 1])
      v = SHAInet::SimpleMatrix.new(4, 2, 0.0)
      v[0, 0] = 1.0; v[0, 1] = 0.0 # write to key 0
      v[1, 0] = 0.0; v[1, 1] = 3.0 # write to key 1
      dst, _ = SHAInet::GatedDeltaNet.recurrent(q, k, v, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0])

      # t=2 queries key 0, t=3 queries key 1. Writing key 1 must not have disturbed key 0.
      dst[2, 0].should be_close(1.0, 1e-9)
      dst[2, 1].should be_close(0.0, 1e-9)
      dst[3, 0].should be_close(0.0, 1e-9)
      dst[3, 1].should be_close(3.0, 1e-9)
    end
  end

  describe "the delta rule, as distinct from accumulation" do
    it "REPLACES the value on a second write to the same key" do
      # This is the sharpest distinction from plain linear attention. Linear attention
      # accumulates, so writing 1.0 then 5.0 to one key would read back 6.0. The delta rule
      # with beta=1 erases the old association first, so it must read back 5.0.
      k = one_hot(3, 4, [0, 0, 0])
      q = one_hot(3, 4, [0, 0, 0])
      v = SHAInet::SimpleMatrix.new(3, 1, 0.0)
      v[0, 0] = 1.0
      v[1, 0] = 5.0
      dst, _ = SHAInet::GatedDeltaNet.recurrent(q, k, v, [1.0, 1.0, 1.0], [1.0, 1.0, 0.0])

      dst[2, 0].should be_close(5.0, 1e-9)
      dst[2, 0].should_not be_close(6.0, 1e-6)
    end

    it "blends old and new by beta on a partial write" do
      # beta=0.5 gives v_new = 0.5*v + 0.5*v_old, per the rule's construction of v_t^new.
      k = one_hot(3, 4, [0, 0, 0])
      q = one_hot(3, 4, [0, 0, 0])
      v = SHAInet::SimpleMatrix.new(3, 1, 0.0)
      v[0, 0] = 1.0
      v[1, 0] = 5.0
      dst, _ = SHAInet::GatedDeltaNet.recurrent(q, k, v, [1.0, 1.0, 1.0], [1.0, 0.5, 0.0])

      dst[2, 0].should be_close(3.0, 1e-9)
    end
  end

  describe "gating" do
    it "erases everything when alpha is zero" do
      # a->0 is the memory-clearing behaviour the gate exists for, and what pure DeltaNet
      # cannot do.
      k = one_hot(3, 4, [0, 1, 0])
      q = one_hot(3, 4, [0, 1, 0])
      v = SHAInet::SimpleMatrix.new(3, 1, 0.0)
      v[0, 0] = 7.0
      dst, state = SHAInet::GatedDeltaNet.recurrent(q, k, v, [1.0, 0.0, 1.0], [1.0, 0.0, 0.0])

      # The alpha=0 token wiped the state, so the key written at t=0 reads back zero.
      dst[2, 0].should be_close(0.0, 1e-9)
      state.rows.times { |i| state.cols.times { |j| state[i, j].should be_close(0.0, 1e-9) } }
    end

    it "decays the state by exactly the cumulative product of alpha when beta is zero" do
      # With no writing, the state is only scaled, so the read is the product of the alphas.
      # This pins the gate's placement: applying alpha to the write term as well would give
      # a different factor here.
      k = one_hot(4, 4, [0, 0, 0, 0])
      q = one_hot(4, 4, [0, 0, 0, 0])
      v = SHAInet::SimpleMatrix.new(4, 1, 0.0)
      v[0, 0] = 1.0
      alphas = [1.0, 0.5, 0.25, 0.5]
      dst, _ = SHAInet::GatedDeltaNet.recurrent(q, k, v, alphas, [1.0, 0.0, 0.0, 0.0])

      # Written at t=0, then decayed by alpha at t=1, t=2, t=3.
      dst[3, 0].should be_close(0.5 * 0.25 * 0.5, 1e-9)
    end

    it "does not decay the value written in the same step" do
      # beta v k^T sits OUTSIDE the alpha factor in Eq. 10, so a fresh write is read back
      # undecayed however small alpha is. Getting this wrong is an easy transcription slip.
      k = one_hot(1, 4, [0])
      q = one_hot(1, 4, [0])
      v = SHAInet::SimpleMatrix.new(1, 1, 0.0)
      v[0, 0] = 2.0
      dst, _ = SHAInet::GatedDeltaNet.recurrent(q, k, v, [0.001], [1.0])

      dst[0, 0].should be_close(2.0, 1e-9)
    end
  end

  describe "state continuation" do
    it "splitting a sequence and carrying the state equals processing it whole" do
      # The property every chunked prefill and every decode step depends on. If this fails,
      # a chunked fast path can never be trusted, so it is specified against the reference
      # BEFORE that path exists.
      rng = Random.new(1234)
      seq, d_k, d_v = 12, 6, 5
      q = SHAInet::SimpleMatrix.new(seq, d_k, 0.0)
      k = SHAInet::SimpleMatrix.new(seq, d_k, 0.0)
      v = SHAInet::SimpleMatrix.new(seq, d_v, 0.0)
      seq.times do |t|
        d_k.times { |j| q[t, j] = rng.rand * 2.0 - 1.0; k[t, j] = rng.rand * 2.0 - 1.0 }
        d_v.times { |j| v[t, j] = rng.rand * 2.0 - 1.0 }
      end
      SHAInet::GatedDeltaNet.l2_normalize!(k)
      alpha = Array(Float64).new(seq) { 0.5 + rng.rand * 0.49 }
      beta = Array(Float64).new(seq) { rng.rand }

      whole, _ = SHAInet::GatedDeltaNet.recurrent(q, k, v, alpha, beta)

      # Same input, processed as 5 tokens then 7, carrying the state across.
      split = 5
      qa = SHAInet::SimpleMatrix.new(split, d_k, 0.0)
      ka = SHAInet::SimpleMatrix.new(split, d_k, 0.0)
      va = SHAInet::SimpleMatrix.new(split, d_v, 0.0)
      split.times do |t|
        d_k.times { |j| qa[t, j] = q[t, j]; ka[t, j] = k[t, j] }
        d_v.times { |j| va[t, j] = v[t, j] }
      end
      rest = seq - split
      qb = SHAInet::SimpleMatrix.new(rest, d_k, 0.0)
      kb = SHAInet::SimpleMatrix.new(rest, d_k, 0.0)
      vb = SHAInet::SimpleMatrix.new(rest, d_v, 0.0)
      rest.times do |t|
        d_k.times { |j| qb[t, j] = q[split + t, j]; kb[t, j] = k[split + t, j] }
        d_v.times { |j| vb[t, j] = v[split + t, j] }
      end

      first, carried = SHAInet::GatedDeltaNet.recurrent(qa, ka, va, alpha[0, split], beta[0, split])
      second, _ = SHAInet::GatedDeltaNet.recurrent(qb, kb, vb, alpha[split, rest], beta[split, rest], carried)

      split.times { |t| d_v.times { |j| first[t, j].should be_close(whole[t, j], 1e-9) } }
      rest.times { |t| d_v.times { |j| second[t, j].should be_close(whole[split + t, j], 1e-9) } }
    end

    it "one token at a time equals the whole sequence, which is what decode does" do
      rng = Random.new(99)
      seq, d_k, d_v = 8, 4, 4
      q = SHAInet::SimpleMatrix.new(seq, d_k, 0.0)
      k = SHAInet::SimpleMatrix.new(seq, d_k, 0.0)
      v = SHAInet::SimpleMatrix.new(seq, d_v, 0.0)
      seq.times do |t|
        d_k.times { |j| q[t, j] = rng.rand; k[t, j] = rng.rand }
        d_v.times { |j| v[t, j] = rng.rand }
      end
      SHAInet::GatedDeltaNet.l2_normalize!(k)
      alpha = Array(Float64).new(seq) { 0.9 }
      beta = Array(Float64).new(seq) { 0.7 }

      whole, _ = SHAInet::GatedDeltaNet.recurrent(q, k, v, alpha, beta)

      state = nil
      seq.times do |t|
        q1 = SHAInet::SimpleMatrix.new(1, d_k, 0.0)
        k1 = SHAInet::SimpleMatrix.new(1, d_k, 0.0)
        v1 = SHAInet::SimpleMatrix.new(1, d_v, 0.0)
        d_k.times { |j| q1[0, j] = q[t, j]; k1[0, j] = k[t, j] }
        d_v.times { |j| v1[0, j] = v[t, j] }
        step, state = SHAInet::GatedDeltaNet.recurrent(q1, k1, v1, [alpha[t]], [beta[t]], state)
        d_v.times { |j| step[0, j].should be_close(whole[t, j], 1e-9) }
      end
    end
  end

  describe "l2_normalize!" do
    it "makes each row unit length" do
      m = SHAInet::SimpleMatrix.new(2, 3, 0.0)
      m[0, 0] = 3.0; m[0, 1] = 4.0; m[0, 2] = 0.0
      m[1, 0] = 1.0; m[1, 1] = 1.0; m[1, 2] = 1.0
      SHAInet::GatedDeltaNet.l2_normalize!(m)

      m.rows.times do |i|
        norm = 0.0
        m.cols.times { |j| norm += m[i, j] * m[i, j] }
        Math.sqrt(norm).should be_close(1.0, 1e-5)
      end
    end

    it "leaves a zero row finite instead of producing NaN" do
      # A short conv followed by SiLU can produce an all-but-zero row, and a NaN there would
      # poison the state for every later token rather than failing visibly.
      m = SHAInet::SimpleMatrix.new(1, 3, 0.0)
      SHAInet::GatedDeltaNet.l2_normalize!(m)
      m.cols.times { |j| m[0, j].nan?.should be_false }
    end
  end

  describe "state size" do
    it "is independent of sequence length, which is the point over a KV cache" do
      # Softmax attention pays 96 KiB per token on the 30B, which is what caps context here.
      # The recurrent state must be the same size whatever the length.
      d_k, d_v = 8, 6
      short_q = SHAInet::SimpleMatrix.new(2, d_k, 0.5)
      short_k = SHAInet::SimpleMatrix.new(2, d_k, 0.5)
      short_v = SHAInet::SimpleMatrix.new(2, d_v, 0.5)
      long_q = SHAInet::SimpleMatrix.new(200, d_k, 0.5)
      long_k = SHAInet::SimpleMatrix.new(200, d_k, 0.5)
      long_v = SHAInet::SimpleMatrix.new(200, d_v, 0.5)

      _, s_short = SHAInet::GatedDeltaNet.recurrent(short_q, short_k, short_v,
        Array(Float64).new(2, 0.9), Array(Float64).new(2, 0.5))
      _, s_long = SHAInet::GatedDeltaNet.recurrent(long_q, long_k, long_v,
        Array(Float64).new(200, 0.9), Array(Float64).new(200, 0.5))

      s_short.rows.should eq(s_long.rows)
      s_short.cols.should eq(s_long.cols)
      (s_long.rows * s_long.cols).should eq(d_v * d_k)
    end
  end
end
