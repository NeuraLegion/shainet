module SHAInet
  class MultiHeadAttention
    getter num_heads, d_model, head_dim
    @head_dim : Int32
    @w_q : SimpleMatrix | CudaMatrix
    @w_k : SimpleMatrix | CudaMatrix
    @w_v : SimpleMatrix | CudaMatrix
    @w_o : SimpleMatrix | CudaMatrix
    @x : SimpleMatrix | CudaMatrix | Nil
    @q_heads : Array(SimpleMatrix | CudaMatrix)
    @k_heads : Array(SimpleMatrix | CudaMatrix)
    @v_heads : Array(SimpleMatrix | CudaMatrix)
    @attn : Array(SimpleMatrix | CudaMatrix)
    @out : SimpleMatrix | CudaMatrix

    # Gradient matrices - keep same type as weights
    @g_w_q : SimpleMatrix | CudaMatrix
    @g_w_k : SimpleMatrix | CudaMatrix
    @g_w_v : SimpleMatrix | CudaMatrix
    @g_w_o : SimpleMatrix | CudaMatrix

    getter w_q, w_k, w_v, w_o
    property g_w_q, g_w_k, g_w_v, g_w_o

    def initialize(@d_model : Int32, @num_heads : Int32)
      @head_dim = (@d_model // @num_heads)
      # Use CudaMatrix when CUDA is available for better performance
      mat_klass = CUDA.fully_available? ? CudaMatrix : SimpleMatrix

      # Use consistent matrix type throughout - no more TensorMatrix mixing
      @w_q = mat_klass.new(@d_model, @d_model).random_fill!
      @w_k = mat_klass.new(@d_model, @d_model).random_fill!
      @w_v = mat_klass.new(@d_model, @d_model).random_fill!
      @w_o = mat_klass.new(@d_model, @d_model).random_fill!

      # Initialize gradients with same type as weights
      @g_w_q = mat_klass.zeros(@d_model, @d_model)
      @g_w_k = mat_klass.zeros(@d_model, @d_model)
      @g_w_v = mat_klass.zeros(@d_model, @d_model)
      @g_w_o = mat_klass.zeros(@d_model, @d_model)

      @q_heads = [] of (SimpleMatrix | CudaMatrix)
      @k_heads = [] of (SimpleMatrix | CudaMatrix)
      @v_heads = [] of (SimpleMatrix | CudaMatrix)
      @attn = [] of (SimpleMatrix | CudaMatrix)
      @out = mat_klass.zeros(1, 1)
    end

    # `x` is expected to have each row representing a batch entry.
    # Sequence length should be encoded along the column dimension or through
    # multiple calls when processing sequences step-by-step.
    def forward(x : SimpleMatrix | CudaMatrix, mask : SimpleMatrix | CudaMatrix | Nil = nil)
      @x = x

      # Compute Q, K, V projections - weights and input should be same type
      q = x * @w_q
      k = x * @w_k
      v = x * @w_v

      @q_heads = [] of (SimpleMatrix | CudaMatrix)
      @k_heads = [] of (SimpleMatrix | CudaMatrix)
      @v_heads = [] of (SimpleMatrix | CudaMatrix)
      @attn = [] of (SimpleMatrix | CudaMatrix)
      outputs = [] of (SimpleMatrix | CudaMatrix)

      # Split into heads and compute attention - all operations stay on same device
      @num_heads.times do |h|
        qs = q.slice_cols(h * @head_dim, @head_dim)
        ks = k.slice_cols(h * @head_dim, @head_dim)
        vs = v.slice_cols(h * @head_dim, @head_dim)

        @q_heads << qs
        @k_heads << ks
        @v_heads << vs

        # Attention computation - stays on GPU if input was GPU
        scores = qs * ks.transpose * (1.0 / Math.sqrt(@head_dim.to_f))

        # Apply mask if provided
        if m = mask
          raise "mask size mismatch" unless m.rows == scores.rows && m.cols == scores.cols
          scores = scores + m
        end

        # GPU-accelerated softmax when available
        attn = SHAInet.softmax_rows(scores)
        @attn << attn

        # Compute output for this head
        outputs << (attn * vs)
      end

      # Concatenate heads - maintain device type
      mat_klass = x.class
      concat = mat_klass.new(x.rows, @d_model)
      @num_heads.times do |h|
        concat.set_cols!(h * @head_dim, outputs[h])
      end

      # Final projection - keeps result on same device
      @out = concat * @w_o
      @out
    end

    # `d_out` should follow the same batch-first layout as the input to
    # `forward`, where each row corresponds to a batch entry.
    def backward(d_out : SimpleMatrix)
      mat_klass = d_out.class
      x = @x.not_nil!

      # Gradient w.r.t. W_o
      concat = mat_klass.new(x.rows, @d_model)
      @num_heads.times do |h|
        output = @attn[h] * @v_heads[h]
        concat.set_cols!(h * @head_dim, output)
      end
      @g_w_o = @g_w_o + (concat.transpose * d_out)

      # Gradient w.r.t. concat
      d_concat = d_out * @w_o.transpose

      # Gradients w.r.t. each head
      d_q_heads = [] of SimpleMatrix
      d_k_heads = [] of SimpleMatrix
      d_v_heads = [] of SimpleMatrix

      @num_heads.times do |h|
        d_out_h = d_concat.slice_cols(h * @head_dim, @head_dim)

        # Gradient w.r.t. V
        d_v = @attn[h].transpose * d_out_h
        d_v_heads << d_v

        # Gradient w.r.t. attention weights
        d_attn = d_out_h * @v_heads[h].transpose

        # Gradient w.r.t. scores (softmax backward)
        d_scores = softmax_backward(d_attn, @attn[h])

        # Scale by 1/sqrt(d_k)
        d_scores = d_scores * (1.0 / Math.sqrt(@head_dim.to_f))

        # Gradients w.r.t. Q and K
        d_q = d_scores * @k_heads[h]
        d_k = d_scores.transpose * @q_heads[h]

        d_q_heads << d_q
        d_k_heads << d_k
      end

      # Concatenate head gradients
      d_q_concat = mat_klass.new(x.rows, @d_model)
      d_k_concat = mat_klass.new(x.rows, @d_model)
      d_v_concat = mat_klass.new(x.rows, @d_model)

      @num_heads.times do |h|
        d_q_concat.set_cols!(h * @head_dim, d_q_heads[h])
        d_k_concat.set_cols!(h * @head_dim, d_k_heads[h])
        d_v_concat.set_cols!(h * @head_dim, d_v_heads[h])
      end

      # Gradients w.r.t. projection weights
      @g_w_q = @g_w_q + (x.transpose * d_q_concat)
      @g_w_k = @g_w_k + (x.transpose * d_k_concat)
      @g_w_v = @g_w_v + (x.transpose * d_v_concat)

      # Gradient w.r.t. input
      d_x = (d_q_concat * @w_q.transpose) +
            (d_k_concat * @w_k.transpose) +
            (d_v_concat * @w_v.transpose)

      d_x
    end

    def apply_gradients(lr : Float64)
      @w_q = @w_q - (@g_w_q * lr)
      @w_k = @w_k - (@g_w_k * lr)
      @w_v = @w_v - (@g_w_v * lr)
      @w_o = @w_o - (@g_w_o * lr)

      # Sync updated weights to device if using CUDA
      if CUDA.fully_available?
        [@w_q, @w_k, @w_v, @w_o].each do |mat|
          if mat.is_a?(CudaMatrix)
            mat.sync_to_device! unless mat.device_dirty?
          end
        end
      end

      # Clear gradients
      zero_gradients
    end

    def zero_gradients
      # Use same matrix type as weights for gradients
      mat_klass = @w_q.class
      @g_w_q = mat_klass.zeros(@d_model, @d_model)
      @g_w_k = mat_klass.zeros(@d_model, @d_model)
      @g_w_v = mat_klass.zeros(@d_model, @d_model)
      @g_w_o = mat_klass.zeros(@d_model, @d_model)
    end

    private def softmax_backward(d_out : SimpleMatrix, softmax_out : SimpleMatrix)
      # Efficient softmax gradient computation
      mat_klass = d_out.class
      result = mat_klass.new(d_out.rows, d_out.cols)

      d_out.rows.times do |i|
        # For each row, compute: softmax * (d_out - sum(softmax * d_out))
        sum = 0.0
        d_out.cols.times { |j| sum += softmax_out[i, j] * d_out[i, j] }

        d_out.cols.times do |j|
          result[i, j] = softmax_out[i, j] * (d_out[i, j] - sum)
        end
      end

      result
    end
  end
end
