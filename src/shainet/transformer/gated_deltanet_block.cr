module SHAInet
  # A Gated DeltaNet token-mixer block: the linear-attention layer type in a qwen3_5 hybrid
  # stack, standing in for LlamaBlock wherever layer_types says "linear_attention".
  #
  # Macro shape follows LlamaBlock exactly, so the two are interchangeable in a stack:
  #
  #   x -> norm1 -> mixer -> +x  ->  norm2 -> ffn -> +
  #
  # Only the mixer differs. Per the paper's block design (arXiv:2412.06464 section 3.4):
  #
  #   q, k: linear proj -> short conv -> SiLU -> L2 normalize
  #   v:    linear proj -> short conv -> SiLU
  #   a, b: linear proj only
  #   out:  gated delta rule -> norm -> SiLU gate -> output proj
  #
  # Their ablation says the short conv and the output gate both matter (removing either costs
  # more than removing the output norm), and that L2 normalization is essential -- so none of
  # those pieces are optional.
  #
  # WHAT THIS BUYS on a 16 GB card: no KV cache. The mixer's whole memory is a [d_v, d_k] state
  # per head plus a kernel-1 conv window, neither of which grows with context. Softmax attention
  # costs 96 KiB per position on the 30B, which is what caps context at 32768 here.
  #
  # UNVERIFIED against a checkpoint. The gate parameterization below is the documented Mamba2
  # form, but the paper says only "we use Mamba2's parameterization for alpha but omit it for
  # brevity", and there is no reference implementation on this machine to diff against. The
  # shapes, the state carry and the prefill/decode equivalence ARE specified and tested; the
  # exact gate formula needs a real Qwen3.5 checkpoint to confirm, which is why the loader still
  # refuses to build one of these from weights.
  class GatedDeltaNetBlock < MatrixLayer
    getter d_model : Int32
    getter num_v_heads : Int32
    getter num_k_heads : Int32
    getter head_k : Int32
    getter head_v : Int32
    getter conv_kernel : Int32

    getter norm1 : RMSNorm
    getter norm2 : RMSNorm
    getter out_norm : RMSNorm
    getter ffn : SwiGLUFF

    getter w_q : SimpleMatrix
    getter w_k : SimpleMatrix
    getter w_v : SimpleMatrix
    getter w_o : SimpleMatrix
    getter w_gate : SimpleMatrix
    getter w_alpha : SimpleMatrix
    getter w_beta : SimpleMatrix
    # Mamba2's per-head decay parameters. a_log is stored in log space because the decay must
    # stay in (0,1) for the state to be stable, and exp(-exp(a_log)*dt) is in (0,1) for any
    # real a_log and any dt > 0 -- no clamping needed.
    getter a_log : Array(Float64)
    getter dt_bias : Array(Float64)

    def initialize(@d_model : Int32, ff_hidden : Int32,
                   @num_v_heads : Int32 = 32, @num_k_heads : Int32 = 16,
                   @head_k : Int32 = 128, @head_v : Int32 = 128,
                   @conv_kernel : Int32 = 4, eps : Float64 = 1e-6)
      super(@d_model, SHAInet.none)
      raise ArgumentError.new("num_v_heads must be positive") unless @num_v_heads > 0
      raise ArgumentError.new("num_k_heads must be positive") unless @num_k_heads > 0
      # Value heads are grouped over key heads, as in grouped-query attention: several value
      # heads share one key head's state dimension.
      raise ArgumentError.new("num_v_heads (#{@num_v_heads}) must be divisible by num_k_heads (#{@num_k_heads})") unless @num_v_heads % @num_k_heads == 0

      @norm1 = RMSNorm.new(@d_model, eps)
      @norm2 = RMSNorm.new(@d_model, eps)
      @out_norm = RMSNorm.new(@num_v_heads * @head_v, eps)
      @ffn = SwiGLUFF.new(@d_model, ff_hidden)

      k_dim = @num_k_heads * @head_k
      v_dim = @num_v_heads * @head_v
      @w_q = SimpleMatrix.new(@d_model, k_dim)
      @w_k = SimpleMatrix.new(@d_model, k_dim)
      @w_v = SimpleMatrix.new(@d_model, v_dim)
      @w_o = SimpleMatrix.new(v_dim, @d_model)
      @w_gate = SimpleMatrix.new(@d_model, v_dim)
      @w_alpha = SimpleMatrix.new(@d_model, @num_v_heads)
      @w_beta = SimpleMatrix.new(@d_model, @num_v_heads)
      @a_log = Array(Float64).new(@num_v_heads, 0.0)
      @dt_bias = Array(Float64).new(@num_v_heads, 0.0)

      @conv_q = ShortConv.new(k_dim, @conv_kernel)
      @conv_k = ShortConv.new(k_dim, @conv_kernel)
      @conv_v = ShortConv.new(v_dim, @conv_kernel)

      @state = Array(SimpleMatrix?).new(@num_v_heads, nil)
      @conv_state_q = nil
      @conv_state_k = nil
      @conv_state_v = nil
    end

    getter conv_q : ShortConv
    getter conv_k : ShortConv
    getter conv_v : ShortConv

    @state : Array(SimpleMatrix?)
    @conv_state_q : SimpleMatrix?
    @conv_state_k : SimpleMatrix?
    @conv_state_v : SimpleMatrix?

    # Drop all recurrent state, so the next forward starts a fresh sequence.
    #
    # The analogue of clearing a KV cache, and it must clear the conv windows TOO: leaving them
    # behind would let the first positions of a new sequence see the tail of the previous one,
    # which is a silent correctness bug rather than a visible failure.
    def clear_cache!
      @num_v_heads.times { |h| @state[h] = nil }
      @conv_state_q = nil
      @conv_state_k = nil
      @conv_state_v = nil
    end

    # Bytes of recurrent state held, for comparison against a KV cache.
    def state_bytes : Int64
      per_head = (@head_v * @head_k * 4).to_i64
      conv = ((@num_k_heads * @head_k * 2 + @num_v_heads * @head_v) * (@conv_kernel - 1) * 4).to_i64
      per_head * @num_v_heads + conv
    end

    # SiLU, x * sigmoid(x).
    private def silu(x : Float64) : Float64
      x / (1.0 + Math.exp(-x))
    end

    private def softplus(x : Float64) : Float64
      # log1p(exp(x)) guarded for large x, where exp overflows and the function is ~x.
      x > 20.0 ? x : Math.log1p(Math.exp(x))
    end

    # Per-head alpha and beta for each position.
    #
    # Mamba2's parameterization, which the paper adopts without restating:
    #
    #   dt    = softplus(x W_alpha + dt_bias)
    #   alpha = exp(-exp(a_log) * dt)      in (0,1) for any real a_log, dt > 0
    #   beta  = sigmoid(x W_beta)          in (0,1)
    #
    # Both ranges are structural rather than clamped, which matters: an alpha at or above 1
    # would let the state grow without bound over a long context, and a beta outside (0,1) would
    # break the delta rule's interpretation as a blend of old and new.
    def gates(normed : SimpleMatrix) : {Array(Array(Float64)), Array(Array(Float64))}
      seq = normed.rows
      alpha = Array(Array(Float64)).new(@num_v_heads) { Array(Float64).new(seq, 0.0) }
      beta = Array(Array(Float64)).new(@num_v_heads) { Array(Float64).new(seq, 0.0) }

      seq.times do |t|
        @num_v_heads.times do |h|
          a_acc = @dt_bias[h]
          b_acc = 0.0
          @d_model.times do |i|
            xv = normed[t, i].to_f64
            a_acc += xv * @w_alpha[i, h].to_f64
            b_acc += xv * @w_beta[i, h].to_f64
          end
          dt = softplus(a_acc)
          alpha[h][t] = Math.exp(-Math.exp(@a_log[h]) * dt)
          beta[h][t] = 1.0 / (1.0 + Math.exp(-b_acc))
        end
      end

      {alpha, beta}
    end

    # The mixer: normed input in, [seq, d_model] out.
    #
    # `chunk` selects the operator form. Above 1 it uses the chunked parallel path for prefill;
    # at 1 it is the sequential recurrence, which is all decode can use anyway. Both produce the
    # same numbers, which is specified rather than assumed.
    def mix(normed : SimpleMatrix, chunk : Int32 = 64) : SimpleMatrix
      seq = normed.rows
      k_dim = @num_k_heads * @head_k
      v_dim = @num_v_heads * @head_v

      q_lin = project(normed, @w_q, k_dim)
      k_lin = project(normed, @w_k, k_dim)
      v_lin = project(normed, @w_v, v_dim)

      q_c, @conv_state_q = @conv_q.forward(q_lin, @conv_state_q)
      k_c, @conv_state_k = @conv_k.forward(k_lin, @conv_state_k)
      v_c, @conv_state_v = @conv_v.forward(v_lin, @conv_state_v)

      seq.times do |t|
        k_dim.times { |j| q_c[t, j] = silu(q_c[t, j].to_f64); k_c[t, j] = silu(k_c[t, j].to_f64) }
        v_dim.times { |j| v_c[t, j] = silu(v_c[t, j].to_f64) }
      end

      alpha, beta = gates(normed)
      heads_per_k = @num_v_heads // @num_k_heads
      mixed = SimpleMatrix.new(seq, v_dim, 0.0)

      @num_v_heads.times do |h|
        kh = h // heads_per_k
        qh = SimpleMatrix.new(seq, @head_k, 0.0)
        khm = SimpleMatrix.new(seq, @head_k, 0.0)
        vh = SimpleMatrix.new(seq, @head_v, 0.0)
        seq.times do |t|
          @head_k.times do |j|
            qh[t, j] = q_c[t, kh * @head_k + j]
            khm[t, j] = k_c[t, kh * @head_k + j]
          end
          @head_v.times { |j| vh[t, j] = v_c[t, h * @head_v + j] }
        end
        # L2 on q and k only, per the block design. The paper's ablation calls it essential, and
        # the recurrence's stability depends on ||k|| = 1.
        GatedDeltaNet.l2_normalize!(qh)
        GatedDeltaNet.l2_normalize!(khm)

        out_h, st = if chunk > 1
                      GatedDeltaNet.chunked(qh, khm, vh, alpha[h], beta[h], @state[h], chunk: chunk)
                    else
                      GatedDeltaNet.recurrent(qh, khm, vh, alpha[h], beta[h], @state[h])
                    end
        @state[h] = st
        seq.times { |t| @head_v.times { |j| mixed[t, h * @head_v + j] = out_h[t, j] } }
      end

      # Output norm, then the SiLU gate, then project back to d_model.
      normed_mix = @out_norm.forward(mixed)
      gate = project(normed, @w_gate, v_dim)
      seq.times do |t|
        v_dim.times { |j| normed_mix[t, j] = normed_mix[t, j].to_f64 * silu(gate[t, j].to_f64) }
      end
      project(normed_mix, @w_o, @d_model)
    end

    private def project(x : SimpleMatrix, w : SimpleMatrix, out_dim : Int32) : SimpleMatrix
      dst = SimpleMatrix.new(x.rows, out_dim, 0.0)
      x.rows.times do |t|
        out_dim.times do |o|
          acc = 0.0
          x.cols.times { |i| acc += x[t, i].to_f64 * w[i, o].to_f64 }
          dst[t, o] = acc
        end
      end
      dst
    end

    def forward(x : SimpleMatrix) : SimpleMatrix
      h = x + mix(@norm1.forward(x))
      h + @ffn.forward(@norm2.forward(h))
    end

    # Single-token step, for decode. Same math, chunk 1.
    def forward_cached(x : SimpleMatrix) : SimpleMatrix
      h = x + mix(@norm1.forward(x), chunk: 1)
      h + @ffn.forward(@norm2.forward(h))
    end

    def backward(d_out : SimpleMatrix) : SimpleMatrix
      raise NotImplementedError.new("GatedDeltaNetBlock is inference-only for now")
    end

    def apply_gradients(lr : Float64)
      raise NotImplementedError.new("GatedDeltaNetBlock is inference-only for now")
    end
  end
end
