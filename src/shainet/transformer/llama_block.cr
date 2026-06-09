require "../basic/matrix_layer"

module SHAInet
  # LLaMA-style transformer block.
  # Pre-norm with RMSNorm, SwiGLU FFN, RoPE in attention, no bias.
  # Supports Grouped Query Attention (GQA) where num_kv_heads < num_heads.
  class LlamaBlock < MatrixLayer
    getter norm1 : RMSNorm # input_layernorm
    getter norm2 : RMSNorm # post_attention_layernorm
    getter ffn : SwiGLUFF
    getter num_heads : Int32
    getter num_kv_heads : Int32
    getter head_dim : Int32
    getter d_model : Int32
    property rope_theta : Float64

    # Attention projections (no bias)
    property w_q : SimpleMatrix | CudaMatrix
    property w_k : SimpleMatrix | CudaMatrix
    property w_v : SimpleMatrix | CudaMatrix
    property w_o : SimpleMatrix | CudaMatrix

    def initialize(@d_model : Int32, @num_heads : Int32, ff_hidden : Int32,
                   eps : Float64 = 1e-6, @rope_theta : Float64 = 10000.0,
                   @num_kv_heads : Int32 = @num_heads)
      super(@d_model, SHAInet.none)
      raise ArgumentError.new("d_model (#{@d_model}) must be divisible by num_heads (#{@num_heads})") unless @d_model % @num_heads == 0
      raise ArgumentError.new("num_heads (#{@num_heads}) must be divisible by num_kv_heads (#{@num_kv_heads})") unless @num_kv_heads > 0 && @num_heads % @num_kv_heads == 0
      @head_dim = @d_model // @num_heads
      kv_dim = @num_kv_heads * @head_dim
      @norm1 = RMSNorm.new(@d_model, eps)
      @norm2 = RMSNorm.new(@d_model, eps)
      @ffn = SwiGLUFF.new(@d_model, ff_hidden)
      @w_q = SimpleMatrix.new(@d_model, @d_model)
      @w_k = SimpleMatrix.new(@d_model, kv_dim)
      @w_v = SimpleMatrix.new(@d_model, kv_dim)
      @w_o = SimpleMatrix.new(@d_model, @d_model)
    end

    def to_gpu!
      return unless CUDA.fully_available?
      @w_q = @w_q.as(SimpleMatrix).to_cuda unless @w_q.is_a?(CudaMatrix)
      @w_k = @w_k.as(SimpleMatrix).to_cuda unless @w_k.is_a?(CudaMatrix)
      @w_v = @w_v.as(SimpleMatrix).to_cuda unless @w_v.is_a?(CudaMatrix)
      @w_o = @w_o.as(SimpleMatrix).to_cuda unless @w_o.is_a?(CudaMatrix)
      @norm1.to_gpu!
      @norm2.to_gpu!
      @ffn.to_gpu!
    end

    def apply_gradients(lr : Float64)
      # Training support placeholder — backward pass not yet implemented
    end

    def backward(d_out : SimpleMatrix) : SimpleMatrix
      raise "LlamaBlock backward pass not yet implemented"
    end

    def backward(d_out : CudaMatrix) : CudaMatrix
      raise "LlamaBlock backward pass not yet implemented"
    end

    # CPU path
    def forward(x : SimpleMatrix) : SimpleMatrix
      normed = @norm1.forward(x)
      attn = self_attention_cpu(normed)
      h = x + attn
      normed2 = @norm2.forward(h)
      ff_out = @ffn.forward(normed2)
      h + ff_out
    end

    # GPU path
    def forward(x : CudaMatrix) : CudaMatrix
      normed = @norm1.forward(x)
      attn = self_attention_gpu(normed)
      h = x + attn
      normed2 = @norm2.forward(h)
      ff_out = @ffn.forward(normed2)
      h + ff_out
    end

    private def self_attention_cpu(x : SimpleMatrix) : SimpleMatrix
      seq_len = x.rows
      head_dim = @head_dim
      num_heads = @num_heads
      num_kv_heads = @num_kv_heads
      heads_per_kv = num_heads // num_kv_heads
      scale = 1.0 / Math.sqrt(head_dim.to_f64)

      q = x * @w_q.as(SimpleMatrix)
      k = x * @w_k.as(SimpleMatrix)
      v = x * @w_v.as(SimpleMatrix)

      output = SimpleMatrix.new(seq_len, @d_model)

      num_heads.times do |h|
        q_col = h * head_dim
        kv_col = (h // heads_per_kv) * head_dim

        q_h = SimpleMatrix.new(seq_len, head_dim)
        k_h = SimpleMatrix.new(seq_len, head_dim)
        v_h = SimpleMatrix.new(seq_len, head_dim)
        seq_len.times do |s|
          head_dim.times do |d|
            q_h[s, d] = q[s, q_col + d]
            k_h[s, d] = k[s, kv_col + d]
            v_h[s, d] = v[s, kv_col + d]
          end
        end

        q_h = RoPE.apply(q_h, 0, @rope_theta)
        k_h = RoPE.apply(k_h, 0, @rope_theta)

        k_t = k_h.transpose
        scores = q_h * k_t

        attn_weights = SimpleMatrix.new(seq_len, seq_len)
        seq_len.times do |i|
          max_val = -Float64::INFINITY
          (0..i).each { |j| sv = scores[i, j] * scale; max_val = sv if sv > max_val }
          exp_sum = 0.0
          (0..i).each do |j|
            e = Math.exp(scores[i, j] * scale - max_val)
            attn_weights[i, j] = e
            exp_sum += e
          end
          (0..i).each { |j| attn_weights[i, j] = attn_weights[i, j] / exp_sum }
        end

        attn_out = attn_weights * v_h
        seq_len.times do |s|
          head_dim.times do |d|
            output[s, q_col + d] = attn_out[s, d]
          end
        end
      end

      output * @w_o.as(SimpleMatrix)
    end

    private def self_attention_gpu(x : CudaMatrix) : CudaMatrix
      seq_len = x.rows
      head_dim = @head_dim
      num_heads = @num_heads
      num_kv_heads = @num_kv_heads
      heads_per_kv = num_heads // num_kv_heads
      scale = 1.0 / Math.sqrt(head_dim.to_f64)

      q = x * @w_q.as(CudaMatrix)
      k = x * @w_k.as(CudaMatrix)
      v = x * @w_v.as(CudaMatrix)

      output = CudaMatrix.new(seq_len, @d_model)

      num_heads.times do |h|
        q_col = h * head_dim
        kv_col = (h // heads_per_kv) * head_dim

        # Slice heads to CPU for per-head RoPE + causal attention
        # (GPU slice_cols only works within thread limits)
        q.sync_from_device!("attn_head_slice") if q.device_dirty?
        k.sync_from_device!("attn_head_slice") if k.device_dirty?
        v.sync_from_device!("attn_head_slice") if v.device_dirty?

        q_h = CudaMatrix.new(seq_len, head_dim)
        k_h = CudaMatrix.new(seq_len, head_dim)
        v_h = CudaMatrix.new(seq_len, head_dim)
        seq_len.times do |s|
          head_dim.times do |d|
            q_h[s, d] = q[s, q_col + d]
            k_h[s, d] = k[s, kv_col + d]
            v_h[s, d] = v[s, kv_col + d]
          end
        end

        # RoPE on CPU then sync
        q_h.sync_from_device!("rope") if q_h.device_dirty?
        k_h.sync_from_device!("rope") if k_h.device_dirty?
        seq_len.times do |pos|
          (head_dim // 2).times do |i|
            freq = 1.0 / (@rope_theta ** (2.0 * i / head_dim))
            angle = pos * freq
            cos_val = Math.cos(angle)
            sin_val = Math.sin(angle)
            x0 = q_h[pos, 2 * i]; x1 = q_h[pos, 2 * i + 1]
            q_h[pos, 2 * i] = x0 * cos_val - x1 * sin_val
            q_h[pos, 2 * i + 1] = x0 * sin_val + x1 * cos_val
            x0 = k_h[pos, 2 * i]; x1 = k_h[pos, 2 * i + 1]
            k_h[pos, 2 * i] = x0 * cos_val - x1 * sin_val
            k_h[pos, 2 * i + 1] = x0 * sin_val + x1 * cos_val
          end
        end
        q_h.sync_to_device!("rope_done")
        k_h.sync_to_device!("rope_done")
        v_h.sync_to_device!("v_head")

        # GPU matmul for scores
        k_t = k_h.transpose
        scores = q_h * k_t # cuBLAS GEMM

        # Causal softmax on CPU (small seq_len × seq_len matrix)
        scores.sync_from_device!("softmax") if scores.device_dirty?
        attn_weights = CudaMatrix.new(seq_len, seq_len)
        seq_len.times do |i|
          max_val = -Float64::INFINITY
          (0..i).each { |j| sv = scores[i, j] * scale; max_val = sv if sv > max_val }
          exp_sum = 0.0
          (0..i).each do |j|
            e = Math.exp(scores[i, j] * scale - max_val)
            attn_weights[i, j] = e
            exp_sum += e
          end
          (0..i).each { |j| attn_weights[i, j] = attn_weights[i, j] / exp_sum }
        end
        attn_weights.sync_to_device!("softmax_done")

        # GPU matmul for attention output
        attn_out = attn_weights * v_h # cuBLAS GEMM

        # Write back to output
        attn_out.sync_from_device!("concat") if attn_out.device_dirty?
        output.sync_from_device!("concat") if output.device_dirty?
        seq_len.times do |s|
          head_dim.times do |d|
            output[s, q_col + d] = attn_out[s, d]
          end
        end
      end

      output.sync_to_device!("attn_concat_done")
      output * @w_o.as(CudaMatrix) # cuBLAS GEMM
    end
  end

  alias LlamaLayer = LlamaBlock
end
