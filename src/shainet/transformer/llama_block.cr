module SHAInet
  # LLaMA-style transformer block.
  # Pre-norm with RMSNorm, SwiGLU FFN, RoPE in attention, no bias.
  class LlamaBlock
    getter norm1 : RMSNorm     # input_layernorm
    getter norm2 : RMSNorm     # post_attention_layernorm
    getter ffn : SwiGLUFF
    getter num_heads : Int32
    getter head_dim : Int32
    getter d_model : Int32
    property rope_theta : Float64

    # Attention projections (no bias)
    property w_q : SimpleMatrix
    property w_k : SimpleMatrix
    property w_v : SimpleMatrix
    property w_o : SimpleMatrix

    def initialize(@d_model : Int32, @num_heads : Int32, ff_hidden : Int32,
                   eps : Float64 = 1e-6, @rope_theta : Float64 = 10000.0)
      @head_dim = @d_model // @num_heads
      @norm1 = RMSNorm.new(@d_model, eps)
      @norm2 = RMSNorm.new(@d_model, eps)
      @ffn = SwiGLUFF.new(@d_model, ff_hidden)
      @w_q = SimpleMatrix.new(@d_model, @d_model)
      @w_k = SimpleMatrix.new(@d_model, @d_model)
      @w_v = SimpleMatrix.new(@d_model, @d_model)
      @w_o = SimpleMatrix.new(@d_model, @d_model)
    end

    def forward(x : SimpleMatrix) : SimpleMatrix
      # Pre-norm attention
      normed = @norm1.forward(x)
      attn = self_attention(normed)
      h = x + attn

      # Pre-norm FFN
      normed2 = @norm2.forward(h)
      ff_out = @ffn.forward(normed2)
      h + ff_out
    end

    private def self_attention(x : SimpleMatrix) : SimpleMatrix
      seq_len = x.rows

      # Project Q, K, V
      q = x * @w_q # [seq, d_model]
      k = x * @w_k
      v = x * @w_v

      # Split into heads and apply RoPE per head
      # Then compute scaled dot-product attention with causal mask
      d_model = @d_model
      num_heads = @num_heads
      head_dim = @head_dim
      scale = 1.0 / Math.sqrt(head_dim.to_f64)

      # Output accumulator
      output = SimpleMatrix.new(seq_len, d_model)

      num_heads.times do |h|
        col_start = h * head_dim

        # Extract head slices
        q_h = SimpleMatrix.new(seq_len, head_dim)
        k_h = SimpleMatrix.new(seq_len, head_dim)
        v_h = SimpleMatrix.new(seq_len, head_dim)
        seq_len.times do |s|
          head_dim.times do |d|
            q_h[s, d] = q[s, col_start + d]
            k_h[s, d] = k[s, col_start + d]
            v_h[s, d] = v[s, col_start + d]
          end
        end

        # Apply RoPE to Q and K
        q_h = RoPE.apply(q_h, 0, @rope_theta)
        k_h = RoPE.apply(k_h, 0, @rope_theta)

        # Scaled dot-product attention with causal mask
        # scores = Q * K^T * scale
        k_t = k_h.transpose
        scores = q_h * k_t

        # Apply scale and causal mask
        attn_weights = SimpleMatrix.new(seq_len, seq_len)
        seq_len.times do |i|
          # Compute softmax for row i (only positions 0..i for causal)
          max_val = -Float64::INFINITY
          (0..i).each { |j| sv = scores[i, j] * scale; max_val = sv if sv > max_val }

          exp_sum = 0.0
          (0..i).each do |j|
            e = Math.exp(scores[i, j] * scale - max_val)
            attn_weights[i, j] = e
            exp_sum += e
          end
          (0..i).each { |j| attn_weights[i, j] = attn_weights[i, j] / exp_sum }
          # positions j > i stay 0 (causal mask)
        end

        # attn_output = weights * V
        attn_out = attn_weights * v_h

        # Write back to output
        seq_len.times do |s|
          head_dim.times do |d|
            output[s, col_start + d] = attn_out[s, d]
          end
        end
      end

      # Output projection
      output * @w_o
    end
  end
end
