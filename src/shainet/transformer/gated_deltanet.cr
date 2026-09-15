module SHAInet
  # Gated DeltaNet: the linear-attention token mixer used by Qwen3.5 / Qwen3.6 / Qwen3-Next.
  #
  # Exists because the Qwen line moved off softmax attention. Those checkpoints declare
  # model_type "qwen3_5" and stack three linear-attention layers for every one full-attention
  # layer, so three quarters of the stack cannot run on the KV-cache attention path at all.
  #
  # The rule (Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta
  # Rule", ICLR 2025, arXiv:2412.06464, Eq. 10):
  #
  #   S_t = S_{t-1} (a_t (I - b_t k_t k_t^T)) + b_t v_t k_t^T
  #   o_t = S_t q_t
  #
  # It unifies two mechanisms this codebase has neither of. `a_t` in (0,1) is Mamba2's decay,
  # which erases the whole state quickly; `b_t` in (0,1) is DeltaNet's write strength, which
  # replaces one key's association without touching the others. a->0 clears memory, a->1
  # reduces to the pure delta rule.
  #
  # The state is what makes this worth having on a 16 GB card: S is [d_v, d_k] per head and
  # does NOT grow with context. Softmax attention pays 96 KiB per token of KV cache on the
  # 30B, which is what caps context at 32k here; a recurrent state is the same size at 1k
  # tokens as at 1M.
  #
  # Two forms live here on purpose:
  #
  #   * `recurrent` is a direct transcription of Eq. 10, one token at a time. It is the
  #     reference: obviously correct by inspection, and the only form usable for decode, where
  #     there is one token and nothing to parallelize.
  #   * `chunked` is the hardware-efficient form for prefill, and is NOT yet implemented here.
  #
  # Everything is fp64 host math for now. The point of this stage is a correct reference to
  # test a fast path against, not speed -- and with no PyTorch on this machine, a reference
  # that is right by construction is the only thing a GPU kernel can later be checked against.
  module GatedDeltaNet
    # One head's worth of the gated delta rule, sequentially.
    #
    # `q`, `k`, `v` are [seq_len, head_dim] with rows in time order. `k` is expected to be
    # L2-normalized by the caller, as the architecture requires (the paper's ablation calls L2
    # normalization essential, and the recurrence's stability depends on ||k|| = 1).
    #
    # `alpha` and `beta` are per-token scalars, length seq_len.
    #
    # `state` is the [d_v, d_k] carry. Pass the previous chunk's state to continue a sequence,
    # or nil to start from zero. It is MUTATED and returned, so decode can keep one allocation
    # per head for the whole generation.
    #
    # Returns {output [seq_len, d_v], state}.
    def self.recurrent(q : SimpleMatrix, k : SimpleMatrix, v : SimpleMatrix,
                       alpha : Array(Float64), beta : Array(Float64),
                       state : SimpleMatrix? = nil) : {SimpleMatrix, SimpleMatrix}
      seq = q.rows
      d_k = k.cols
      d_v = v.cols
      raise ArgumentError.new("q/k rows disagree: #{q.rows} vs #{k.rows}") unless k.rows == seq
      raise ArgumentError.new("q/v rows disagree: #{q.rows} vs #{v.rows}") unless v.rows == seq
      raise ArgumentError.new("q/k dims disagree: #{q.cols} vs #{d_k}") unless q.cols == d_k
      raise ArgumentError.new("alpha size #{alpha.size} != seq #{seq}") unless alpha.size == seq
      raise ArgumentError.new("beta size #{beta.size} != seq #{seq}") unless beta.size == seq

      s = state || SimpleMatrix.new(d_v, d_k, 0.0)
      raise ArgumentError.new("state must be [#{d_v}, #{d_k}], got [#{s.rows}, #{s.cols}]") unless s.rows == d_v && s.cols == d_k

      dst = SimpleMatrix.new(seq, d_v, 0.0)
      sk = Array(Float64).new(d_v, 0.0)

      seq.times do |t|
        a = alpha[t]
        b = beta[t]

        # sk = S k_t, the value currently associated with this key. Computing it BEFORE the
        # update is what makes this the delta rule rather than a plain accumulation: the old
        # association is what gets removed.
        d_v.times do |i|
          acc = 0.0
          d_k.times { |j| acc += s[i, j] * k[t, j] }
          sk[i] = acc
        end

        # S <- a * (S - b * sk k^T) + b * v_t k^T
        #
        # Expanded from S(a(I - b k k^T)) + b v k^T. Fused into one pass over the state so it
        # is touched once rather than three times; at 32 heads x [128, 128] that difference is
        # the whole cost of the layer.
        d_v.times do |i|
          bv = b * v[t, i]
          ask = a * b * sk[i]
          d_k.times do |j|
            kj = k[t, j]
            s[i, j] = a * s[i, j] - ask * kj + bv * kj
          end
        end

        # o_t = S_t q_t, read AFTER the write, so a token can retrieve what it just stored.
        d_v.times do |i|
          acc = 0.0
          d_k.times { |j| acc += s[i, j] * q[t, j] }
          dst[t, i] = acc
        end
      end

      {dst, s}
    end

    # One head's worth of the gated delta rule, a whole chunk at a time.
    #
    # Mathematically identical to `recurrent` but reorganized into matrix products, which is
    # what makes prefill viable: the sequential form is a dependency chain of length seq_len
    # doing tiny per-step work, while this is a handful of [C, C] and [C, d] products per
    # chunk.
    #
    # From arXiv:2412.06464 section 3.3, with the WY representation extended by the gating
    # terms (their appendix A). Writing g^r for the cumulative product of alpha within the
    # chunk, and G for the decay matrix G_ij = g^i/g^j when i >= j and 0 otherwise:
    #
    #   T = [I + strictLower(diag(b) (G .* K K^T))]^-1 diag(b)
    #   U = T V,   W = T K
    #   Z = U - (g .* W) S^T
    #   O = (g .* Q) S^T + ((Q K^T) .* G) Z
    #   S_new = g^C S + Z^T ((g^C / g) .* K)
    #
    # NOTE the second term of O uses the DECAY-AWARE mask G, not the plain causal mask M that
    # section 3.3 prints. G already zeroes the upper triangle, so it subsumes M, and the decay
    # is required: expanding the recurrence gives G_r q_r = sum_i (g^r/g^i) u_i (k_i . q_r),
    # which is exactly a G-weighted sum, and the correction term agrees too because
    # G_ri (g^i w_i) = g^r w_i. With a plain M the intra-chunk contributions would carry no
    # decay and the two forms would disagree for any alpha < 1 -- which is precisely what the
    # equivalence spec against `recurrent` catches.
    #
    # Arguments and the state carry match `recurrent` exactly, so the two are drop-in
    # substitutable and can be diffed directly. That is the entire reason `recurrent` exists.
    def self.chunked(q : SimpleMatrix, k : SimpleMatrix, v : SimpleMatrix,
                     alpha : Array(Float64), beta : Array(Float64),
                     state : SimpleMatrix? = nil, chunk : Int32 = 64) : {SimpleMatrix, SimpleMatrix}
      seq = q.rows
      d_k = k.cols
      d_v = v.cols
      raise ArgumentError.new("chunk must be positive, got #{chunk}") unless chunk > 0
      raise ArgumentError.new("alpha size #{alpha.size} != seq #{seq}") unless alpha.size == seq
      raise ArgumentError.new("beta size #{beta.size} != seq #{seq}") unless beta.size == seq

      s = state || SimpleMatrix.new(d_v, d_k, 0.0)
      raise ArgumentError.new("state must be [#{d_v}, #{d_k}], got [#{s.rows}, #{s.cols}]") unless s.rows == d_v && s.cols == d_k

      dst = SimpleMatrix.new(seq, d_v, 0.0)

      base = 0
      while base < seq
        c = Math.min(chunk, seq - base)

        # g[r] is the decay from the chunk start through position r inclusive.
        g = Array(Float64).new(c, 1.0)
        run = 1.0
        c.times do |r|
          run *= alpha[base + r]
          g[r] = run
        end
        g_last = run

        kk = Array(Array(Float64)).new(c) { Array(Float64).new(c, 0.0) }
        c.times do |i|
          (i + 1).times do |j|
            acc = 0.0
            d_k.times { |d| acc += k[base + i, d] * k[base + j, d] }
            kk[i][j] = acc
          end
        end

        # Solve for U and W by forward substitution. Exact for a unit lower triangular system,
        # and avoids forming an inverse. Rows are produced in order so one pass yields both.
        #
        # U and W use DIFFERENT coefficients, which is the subtle part.
        #
        # W represents P_r = prod_i (I - b_i k_i k_i^T), the ungated Householder product, whose
        # WY vectors use the PLAIN Gram matrix. The gating factors out of it entirely, because
        # F_r = prod_i a_i (I - b_i k_i k_i^T) = (prod_i a_i) prod_i (I - b_i k_i k_i^T) --
        # the alphas are scalars, so they leave as g^r and P_r is left ungated.
        #
        # U carries the writes and DOES need the decay, since a value written at j must be
        # decayed by g^i/g^j to reach position i (appendix A).
        #
        # Weighting W like U was a real bug here, and one that hides from cold: W is only ever
        # used against the incoming state, so a single chunk starting from S = 0 gives the
        # right answer either way. It showed up only as multi-token chunks CARRYING state --
        # measured 1.4e-2 error at chunk 8 against 6e-8 at chunk 1.
        u = Array(Array(Float64)).new(c) { Array(Float64).new(d_v, 0.0) }
        w = Array(Array(Float64)).new(c) { Array(Float64).new(d_k, 0.0) }
        c.times do |i|
          b = beta[base + i]
          d_v.times { |d| u[i][d] = b * v[base + i, d] }
          d_k.times { |d| w[i][d] = b * k[base + i, d] }
          i.times do |j|
            gram = b * kk[i][j]
            next if gram == 0.0
            decayed = gram * (g[i] / g[j])
            d_v.times { |d| u[i][d] -= decayed * u[j][d] }
            d_k.times { |d| w[i][d] -= gram * w[j][d] }
          end
        end

        # Z = U - (g .* W) S^T: the chunk's writes with the incoming state's own contribution
        # already subtracted, so the same Z serves both the output and the new state.
        z = Array(Array(Float64)).new(c) { Array(Float64).new(d_v, 0.0) }
        c.times do |i|
          d_v.times do |dv|
            acc = 0.0
            d_k.times { |dk| acc += w[i][dk] * s[dv, dk] }
            z[i][dv] = u[i][dv] - g[i] * acc
          end
        end

        # O = (g .* Q) S^T + ((Q K^T) .* G) Z
        c.times do |r|
          d_v.times do |dv|
            acc = 0.0
            d_k.times { |dk| acc += q[base + r, dk] * s[dv, dk] }
            dst[base + r, dv] = g[r] * acc
          end
          (r + 1).times do |i|
            qk = 0.0
            d_k.times { |d| qk += q[base + r, d] * k[base + i, d] }
            coef = qk * (g[r] / g[i])
            next if coef == 0.0
            d_v.times { |dv| dst[base + r, dv] += coef * z[i][dv] }
          end
        end

        # S_new = g^C S + Z^T ((g^C / g) .* K)
        d_v.times do |dv|
          d_k.times do |dk|
            acc = g_last * s[dv, dk]
            c.times { |i| acc += z[i][dv] * (g_last / g[i]) * k[base + i, dk] }
            s[dv, dk] = acc
          end
        end

        base += c
      end

      {dst, s}
    end

    # L2-normalize each row in place, as the q/k paths require.
    #
    # Guards a zero row rather than dividing by zero: a short convolution followed by SiLU can
    # legitimately produce an all-but-zero row, and a NaN there would silently poison the
    # state for every later token instead of failing loudly.
    def self.l2_normalize!(m : SimpleMatrix, eps : Float64 = 1e-6) : SimpleMatrix
      m.rows.times do |i|
        sum = 0.0
        m.cols.times { |j| sum += m[i, j] * m[i, j] }
        inv = 1.0 / Math.sqrt(sum + eps)
        m.cols.times { |j| m[i, j] = m[i, j] * inv }
      end
      m
    end
  end
end
