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
