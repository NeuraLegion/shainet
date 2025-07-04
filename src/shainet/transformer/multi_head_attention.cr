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

    # Convert all internal matrices to GPU
    def to_gpu!
      if CUDA.fully_available?
        @w_q = @w_q.as(SimpleMatrix).to_cuda unless @w_q.is_a?(CudaMatrix)
        @w_k = @w_k.as(SimpleMatrix).to_cuda unless @w_k.is_a?(CudaMatrix)
        @w_v = @w_v.as(SimpleMatrix).to_cuda unless @w_v.is_a?(CudaMatrix)
        @w_o = @w_o.as(SimpleMatrix).to_cuda unless @w_o.is_a?(CudaMatrix)

        @g_w_q = @g_w_q.as(SimpleMatrix).to_cuda unless @g_w_q.is_a?(CudaMatrix)
        @g_w_k = @g_w_k.as(SimpleMatrix).to_cuda unless @g_w_k.is_a?(CudaMatrix)
        @g_w_v = @g_w_v.as(SimpleMatrix).to_cuda unless @g_w_v.is_a?(CudaMatrix)
        @g_w_o = @g_w_o.as(SimpleMatrix).to_cuda unless @g_w_o.is_a?(CudaMatrix)

        @out = @out.as(SimpleMatrix).to_cuda if @out && !@out.is_a?(CudaMatrix)
        @x = @x.as(SimpleMatrix).to_cuda if @x && !@x.is_a?(CudaMatrix)

        # Convert stored head matrices to GPU
        @q_heads = @q_heads.map { |h| h.is_a?(CudaMatrix) ? h.as(SimpleMatrix | CudaMatrix) : h.as(SimpleMatrix).to_cuda.as(SimpleMatrix | CudaMatrix) }
        @k_heads = @k_heads.map { |h| h.is_a?(CudaMatrix) ? h.as(SimpleMatrix | CudaMatrix) : h.as(SimpleMatrix).to_cuda.as(SimpleMatrix | CudaMatrix) }
        @v_heads = @v_heads.map { |h| h.is_a?(CudaMatrix) ? h.as(SimpleMatrix | CudaMatrix) : h.as(SimpleMatrix).to_cuda.as(SimpleMatrix | CudaMatrix) }
        @attn = @attn.map { |h| h.is_a?(CudaMatrix) ? h.as(SimpleMatrix | CudaMatrix) : h.as(SimpleMatrix).to_cuda.as(SimpleMatrix | CudaMatrix) }
      end
    end

    # GPU path - all operations with CudaMatrix
    def forward(x : CudaMatrix, mask : CudaMatrix | Nil = nil) : CudaMatrix
      @x = x

      # Compute Q, K, V projections - GPU path
      q = x * @w_q.as(CudaMatrix)
      k = x * @w_k.as(CudaMatrix)
      v = x * @w_v.as(CudaMatrix)

      @q_heads = Array(SimpleMatrix | CudaMatrix).new
      @k_heads = Array(SimpleMatrix | CudaMatrix).new
      @v_heads = Array(SimpleMatrix | CudaMatrix).new
      @attn = Array(SimpleMatrix | CudaMatrix).new
      outputs = [] of CudaMatrix

      # Split into heads and compute attention - all GPU operations
      @num_heads.times do |h|
        qs = q.slice_cols(h * @head_dim, @head_dim)
        ks = k.slice_cols(h * @head_dim, @head_dim)
        vs = v.slice_cols(h * @head_dim, @head_dim)

        @q_heads << qs
        @k_heads << ks
        @v_heads << vs

        # Attention computation - all GPU
        ks_transposed = ks.transpose
        scores = qs * ks_transposed * (1.0 / Math.sqrt(@head_dim.to_f))

        # Apply mask if provided
        if m = mask
          raise "mask size mismatch" unless m.rows == scores.rows && m.cols == scores.cols
          scores = scores + m
        end

        # GPU-accelerated softmax - ensure scores is CudaMatrix
        scores_cuda = scores.is_a?(CudaMatrix) ? scores : scores.to_cuda
        attn = SHAInet.softmax_rows(scores_cuda)
        @attn << attn

        # Compute output for this head
        outputs << (attn * vs)
      end

      # Concatenate heads - GPU
      concat = CudaMatrix.new(x.rows, @d_model)
      @num_heads.times do |h|
        concat.set_cols!(h * @head_dim, outputs[h])
      end

      # Final projection - GPU
      @out = concat * @w_o.as(CudaMatrix)
      @out.as(CudaMatrix)
    end

    # CPU path - all operations with SimpleMatrix
    def forward(x : SimpleMatrix, mask : SimpleMatrix | Nil = nil) : SimpleMatrix
      @x = x

      # Compute Q, K, V projections - CPU path
      q = x * @w_q.as(SimpleMatrix)
      k = x * @w_k.as(SimpleMatrix)
      v = x * @w_v.as(SimpleMatrix)

      @q_heads = Array(SimpleMatrix | CudaMatrix).new
      @k_heads = Array(SimpleMatrix | CudaMatrix).new
      @v_heads = Array(SimpleMatrix | CudaMatrix).new
      @attn = Array(SimpleMatrix | CudaMatrix).new
      outputs = [] of SimpleMatrix

      # Split into heads and compute attention - all CPU operations
      @num_heads.times do |h|
        qs = q.slice_cols(h * @head_dim, @head_dim)
        ks = k.slice_cols(h * @head_dim, @head_dim)
        vs = v.slice_cols(h * @head_dim, @head_dim)

        @q_heads << qs
        @k_heads << ks
        @v_heads << vs

        # Attention computation - all CPU
        ks_transposed = ks.transpose
        scores = qs * ks_transposed * (1.0 / Math.sqrt(@head_dim.to_f))

        # Apply mask if provided
        if m = mask
          raise "mask size mismatch" unless m.rows == scores.rows && m.cols == scores.cols
          scores = scores + m
        end

        # CPU softmax
        attn = SHAInet.softmax_rows(scores)
        @attn << attn

        # Compute output for this head
        outputs << (attn * vs)
      end

      # Concatenate heads - CPU
      concat = SimpleMatrix.new(x.rows, @d_model)
      @num_heads.times do |h|
        concat.set_cols!(h * @head_dim, outputs[h])
      end

      # Final projection - CPU
      @out = concat * @w_o.as(SimpleMatrix)
      @out.as(SimpleMatrix)
    end

    # `d_out` should follow the same batch-first layout as the input to
    # `forward`, where each row corresponds to a batch entry.
    def backward(d_out : SimpleMatrix)
      mat_klass = d_out.class
      x = @x.not_nil!

      # Gradient w.r.t. W_o
      concat = mat_klass.new(x.rows, @d_model)
      @num_heads.times do |h|
        output = @attn[h].as(SimpleMatrix) * @v_heads[h].as(SimpleMatrix)
        concat.set_cols!(h * @head_dim, output)
      end
      @g_w_o = @g_w_o.as(SimpleMatrix) + (concat.transpose * d_out)

      # Gradient w.r.t. concat
      d_concat = d_out * @w_o.transpose.as(SimpleMatrix)

      # Gradients w.r.t. each head
      d_q_heads = [] of SimpleMatrix
      d_k_heads = [] of SimpleMatrix
      d_v_heads = [] of SimpleMatrix

      @num_heads.times do |h|
        d_out_h = d_concat.slice_cols(h * @head_dim, @head_dim)

        # Gradient w.r.t. V
        d_v = @attn[h].as(SimpleMatrix).transpose * d_out_h
        d_v_heads << d_v

        # Gradient w.r.t. attention weights
        d_attn = d_out_h * @v_heads[h].as(SimpleMatrix).transpose

        # Gradient w.r.t. scores (softmax backward)
        d_scores = softmax_backward(d_attn, @attn[h].as(SimpleMatrix))

        # Scale by 1/sqrt(d_k)
        d_scores = d_scores * (1.0 / Math.sqrt(@head_dim.to_f))

        # Gradients w.r.t. Q and K
        d_q = d_scores * @k_heads[h].as(SimpleMatrix)
        d_k = d_scores.transpose * @q_heads[h].as(SimpleMatrix)

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
      @g_w_q = @g_w_q.as(SimpleMatrix) + (x.as(SimpleMatrix).transpose * d_q_concat)
      @g_w_k = @g_w_k.as(SimpleMatrix) + (x.as(SimpleMatrix).transpose * d_k_concat)
      @g_w_v = @g_w_v.as(SimpleMatrix) + (x.as(SimpleMatrix).transpose * d_v_concat)

      # Gradient w.r.t. input
      d_x = (d_q_concat * @w_q.as(SimpleMatrix).transpose) +
            (d_k_concat * @w_k.as(SimpleMatrix).transpose) +
            (d_v_concat * @w_v.as(SimpleMatrix).transpose)

      d_x
    end

    # GPU path backward - all CudaMatrix operations
    def backward(d_out : CudaMatrix) : CudaMatrix
      x = @x.not_nil!.as(CudaMatrix)

      # Gradient w.r.t. W_o
      concat = CudaMatrix.new(x.rows, @d_model)
      @num_heads.times do |h|
        output = @attn[h].as(CudaMatrix) * @v_heads[h].as(CudaMatrix)
        concat.set_cols!(h * @head_dim, output)
      end
      @g_w_o = @g_w_o.as(CudaMatrix) + (concat.transpose * d_out)

      # Gradient w.r.t. concat
      d_concat = d_out * @w_o.as(CudaMatrix).transpose

      # Gradients w.r.t. each head
      d_q_heads = [] of CudaMatrix
      d_k_heads = [] of CudaMatrix
      d_v_heads = [] of CudaMatrix

      @num_heads.times do |h|
        d_out_h = d_concat.slice_cols(h * @head_dim, @head_dim)

        # Gradient w.r.t. V
        d_v = @attn[h].as(CudaMatrix).transpose * d_out_h
        d_v_heads << d_v

        # Gradient w.r.t. attention weights
        d_attn = d_out_h * @v_heads[h].as(CudaMatrix).transpose

        # Gradient w.r.t. scores (softmax backward)
        d_scores = softmax_backward(d_attn, @attn[h].as(CudaMatrix))

        # Scale by 1/sqrt(d_k)
        d_scores = d_scores * (1.0 / Math.sqrt(@head_dim.to_f))

        # Gradients w.r.t. Q and K
        d_q = d_scores * @k_heads[h].as(CudaMatrix)
        d_k = d_scores.transpose * @q_heads[h].as(CudaMatrix)

        d_q_heads << d_q
        d_k_heads << d_k
      end

      # Concatenate head gradients
      d_q_concat = CudaMatrix.new(x.rows, @d_model)
      d_k_concat = CudaMatrix.new(x.rows, @d_model)
      d_v_concat = CudaMatrix.new(x.rows, @d_model)

      @num_heads.times do |h|
        d_q_concat.set_cols!(h * @head_dim, d_q_heads[h])
        d_k_concat.set_cols!(h * @head_dim, d_k_heads[h])
        d_v_concat.set_cols!(h * @head_dim, d_v_heads[h])
      end

      # Gradients w.r.t. projection weights
      @g_w_q = @g_w_q.as(CudaMatrix) + (x.transpose * d_q_concat)
      @g_w_k = @g_w_k.as(CudaMatrix) + (x.transpose * d_k_concat)
      @g_w_v = @g_w_v.as(CudaMatrix) + (x.transpose * d_v_concat)

      # Gradient w.r.t. input
      d_x = (d_q_concat * @w_q.as(CudaMatrix).transpose) +
            (d_k_concat * @w_k.as(CudaMatrix).transpose) +
            (d_v_concat * @w_v.as(CudaMatrix).transpose)

      d_x
    end

    # GPU path for applying gradients
    def apply_gradients(lr : Float64, device : CudaMatrix.class)
      @w_q = @w_q.as(CudaMatrix) - (@g_w_q.as(CudaMatrix) * lr)
      @w_k = @w_k.as(CudaMatrix) - (@g_w_k.as(CudaMatrix) * lr)
      @w_v = @w_v.as(CudaMatrix) - (@g_w_v.as(CudaMatrix) * lr)
      @w_o = @w_o.as(CudaMatrix) - (@g_w_o.as(CudaMatrix) * lr)

      # Mark updated weights as dirty on device
      @w_q.as(CudaMatrix).mark_device_dirty!
      @w_k.as(CudaMatrix).mark_device_dirty!
      @w_v.as(CudaMatrix).mark_device_dirty!
      @w_o.as(CudaMatrix).mark_device_dirty!

      # Clear gradients
      zero_gradients(CudaMatrix)
    end

    # CPU path for applying gradients
    def apply_gradients(lr : Float64, device : SimpleMatrix.class)
      @w_q = @w_q.as(SimpleMatrix) - (@g_w_q.as(SimpleMatrix) * lr)
      @w_k = @w_k.as(SimpleMatrix) - (@g_w_k.as(SimpleMatrix) * lr)
      @w_v = @w_v.as(SimpleMatrix) - (@g_w_v.as(SimpleMatrix) * lr)
      @w_o = @w_o.as(SimpleMatrix) - (@g_w_o.as(SimpleMatrix) * lr)

      # Clear gradients
      zero_gradients(SimpleMatrix)
    end

    # GPU path for zeroing gradients
    def zero_gradients(device : CudaMatrix.class)
      @g_w_q = CudaMatrix.zeros(@d_model, @d_model)
      @g_w_k = CudaMatrix.zeros(@d_model, @d_model)
      @g_w_v = CudaMatrix.zeros(@d_model, @d_model)
      @g_w_o = CudaMatrix.zeros(@d_model, @d_model)
    end

    # CPU path for zeroing gradients
    def zero_gradients(device : SimpleMatrix.class)
      @g_w_q = SimpleMatrix.zeros(@d_model, @d_model)
      @g_w_k = SimpleMatrix.zeros(@d_model, @d_model)
      @g_w_v = SimpleMatrix.zeros(@d_model, @d_model)
      @g_w_o = SimpleMatrix.zeros(@d_model, @d_model)
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

    # GPU version of softmax backward
    private def softmax_backward(d_out : CudaMatrix, softmax_out : CudaMatrix) : CudaMatrix
      # Use GPU kernel for softmax backward if available
      if CUDA.fully_available?
        begin
          result = CudaMatrix.new(d_out.rows, d_out.cols)
          # Use CUDA kernel for softmax backward pass
          CUDA.softmax_backward(result.device_ptr.not_nil!, d_out.device_ptr.not_nil!, softmax_out.device_ptr.not_nil!, d_out.rows, d_out.cols)
          result.mark_device_dirty!
          return result
        rescue e : Exception
          # Fall back to CPU computation if CUDA fails
        end
      end

      # CPU fallback - sync matrices to host first
      d_out.sync_from_device!
      softmax_out.sync_from_device!

      # Efficient softmax gradient computation on CPU
      result = CudaMatrix.new(d_out.rows, d_out.cols)

      d_out.rows.times do |i|
        # For each row, compute: softmax * (d_out - sum(softmax * d_out))
        sum = 0.0
        d_out.cols.times { |j| sum += softmax_out[i, j] * d_out[i, j] }

        d_out.cols.times do |j|
          result[i, j] = softmax_out[i, j] * (d_out[i, j] - sum)
        end
      end

      result.sync_to_device!
      result
    end
  end
end
