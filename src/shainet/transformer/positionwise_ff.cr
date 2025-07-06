module SHAInet
  class PositionWiseFF
    getter w1, b1, w2, b2
    @w1 : SimpleMatrix | CudaMatrix
    @b1 : SimpleMatrix | CudaMatrix
    @w2 : SimpleMatrix | CudaMatrix
    @b2 : SimpleMatrix | CudaMatrix
    @g_w1 : SimpleMatrix | CudaMatrix
    @g_w2 : SimpleMatrix | CudaMatrix
    @g_b1 : SimpleMatrix | CudaMatrix
    @g_b2 : SimpleMatrix | CudaMatrix
    @x : SimpleMatrix | CudaMatrix | Nil
    @h : SimpleMatrix | CudaMatrix
    @out : SimpleMatrix | CudaMatrix

    # Workspace matrices to avoid repeated allocations
    @workspace_temp_bias : CudaMatrix | Nil = nil

    property g_w1 : SimpleMatrix | CudaMatrix
    property g_w2 : SimpleMatrix | CudaMatrix
    property g_b1 : SimpleMatrix | CudaMatrix
    property g_b2 : SimpleMatrix | CudaMatrix

    def initialize(d_model : Int32, hidden_dim : Int32)
      # Use CudaMatrix when CUDA is available for better performance
      mat_klass = CUDA.fully_available? ? CudaMatrix : SimpleMatrix
      @w1 = mat_klass.new(d_model, hidden_dim).random_fill!
      @b1 = mat_klass.new(1, hidden_dim).random_fill!
      @w2 = mat_klass.new(hidden_dim, d_model).random_fill!
      @b2 = mat_klass.new(1, d_model).random_fill!
      @g_w1 = mat_klass.zeros(d_model, hidden_dim)
      @g_w2 = mat_klass.zeros(hidden_dim, d_model)
      @g_b1 = mat_klass.zeros(1, hidden_dim)
      @g_b2 = mat_klass.zeros(1, d_model)
      @h = mat_klass.zeros(1, 1)
      @out = mat_klass.zeros(1, 1)
    end

    # Convert all internal matrices to GPU
    def to_gpu!
      if CUDA.fully_available?
        @w1 = @w1.as(SimpleMatrix).to_cuda unless @w1.is_a?(CudaMatrix)
        @b1 = @b1.as(SimpleMatrix).to_cuda unless @b1.is_a?(CudaMatrix)
        @w2 = @w2.as(SimpleMatrix).to_cuda unless @w2.is_a?(CudaMatrix)
        @b2 = @b2.as(SimpleMatrix).to_cuda unless @b2.is_a?(CudaMatrix)
        @g_w1 = @g_w1.as(SimpleMatrix).to_cuda unless @g_w1.is_a?(CudaMatrix)
        @g_w2 = @g_w2.as(SimpleMatrix).to_cuda unless @g_w2.is_a?(CudaMatrix)
        @g_b1 = @g_b1.as(SimpleMatrix).to_cuda unless @g_b1.is_a?(CudaMatrix)
        @g_b2 = @g_b2.as(SimpleMatrix).to_cuda unless @g_b2.is_a?(CudaMatrix)
        @h = @h.as(SimpleMatrix).to_cuda if @h && !@h.is_a?(CudaMatrix)
        @out = @out.as(SimpleMatrix).to_cuda if @out && !@out.is_a?(CudaMatrix)
        @x = @x.as(SimpleMatrix).to_cuda if @x && !@x.is_a?(CudaMatrix)
      end
    end

    # GPU path - all CudaMatrix operations with cuDNN optimization
    def forward(x : CudaMatrix) : CudaMatrix
      @x = x
      # Weights are already CudaMatrix in GPU path
      w1_gpu = @w1.as(CudaMatrix)
      b1_gpu = @b1.as(CudaMatrix)
      w2_gpu = @w2.as(CudaMatrix)
      b2_gpu = @b2.as(CudaMatrix)

      @h = x * w1_gpu

      # Use cuDNN for optimized bias addition and ReLU if available
      if CUDNN.available?
        CUDNN.add_bias!(@h.as(CudaMatrix), b1_gpu)
        CUDNN.relu_forward(@h.as(CudaMatrix), @h.as(CudaMatrix))
      else
        @h.as(CudaMatrix).add_bias!(b1_gpu)
        @h.as(CudaMatrix).relu!
      end

      @out = @h.as(CudaMatrix) * w2_gpu

      # Use cuDNN for bias addition if available
      if CUDNN.available?
        CUDNN.add_bias!(@out.as(CudaMatrix), b2_gpu)
      else
        @out.as(CudaMatrix).add_bias!(b2_gpu)
      end

      @out.as(CudaMatrix)
    end

    # CPU path - all SimpleMatrix operations
    def forward(x : SimpleMatrix) : SimpleMatrix
      @x = x
      # Use CPU weights directly (they should be SimpleMatrix already)
      w1_cpu = @w1.as(SimpleMatrix)
      b1_cpu = @b1.as(SimpleMatrix)
      w2_cpu = @w2.as(SimpleMatrix)
      b2_cpu = @b2.as(SimpleMatrix)

      @h = x * w1_cpu
      @h.as(SimpleMatrix).add_bias!(b1_cpu)
      @h.as(SimpleMatrix).relu!
      @out = @h.as(SimpleMatrix) * w2_cpu
      @out.as(SimpleMatrix).add_bias!(b2_cpu)
      @out.as(SimpleMatrix)
    end

    # GPU path backward
    def backward(d_out : CudaMatrix) : CudaMatrix
      w2_gpu = @w2.as(CudaMatrix)
      w2_t = w2_gpu.transpose
      dh = CudaMatrix.get_workspace(d_out.rows, w2_gpu.rows, "pw_dh")
      dh.gemm!(d_out, w2_t)
      CudaMatrix.return_workspace(w2_t)

      # Use in-place gradient accumulation to avoid creating new matrices
      temp_grad_w2 = CudaMatrix.get_workspace(@h.cols, d_out.cols, "pw_grad_w2")
      h_t = @h.as(CudaMatrix).transpose
      temp_grad_w2.gemm!(h_t, d_out)
      @g_w2.as(CudaMatrix).add!(temp_grad_w2)
      CudaMatrix.return_workspace(temp_grad_w2)
      CudaMatrix.return_workspace(h_t)

      # Efficient bias gradient using GPU
      # Use optimized bias gradient accumulation
      accumulate_bias_gradient(@g_b2, d_out)

      drelu = relu_grad(@h.as(CudaMatrix), dh)
      CudaMatrix.return_workspace(dh)

      # Use in-place gradient accumulation to avoid creating new matrices
      temp_grad_w1 = CudaMatrix.get_workspace(@x.not_nil!.cols, drelu.cols, "pw_grad_w1")
      x_t = @x.not_nil!.as(CudaMatrix).transpose
      temp_grad_w1.gemm!(x_t, drelu)
      @g_w1.as(CudaMatrix).add!(temp_grad_w1)
      CudaMatrix.return_workspace(temp_grad_w1)
      CudaMatrix.return_workspace(x_t)

      # Use optimized bias gradient accumulation
      accumulate_bias_gradient(@g_b1, drelu)

      w1_gpu = @w1.as(CudaMatrix)
      w1_t = w1_gpu.transpose
      d_input = CudaMatrix.get_workspace(drelu.rows, w1_gpu.rows, "pw_d_input")
      d_input.gemm!(drelu, w1_t)
      CudaMatrix.return_workspace(w1_t)
      d_input
    end

    # CPU path backward
    def backward(d_out : SimpleMatrix) : SimpleMatrix
      w2_cpu = @w2.as(SimpleMatrix)
      dh = d_out * w2_cpu.transpose

      # For SimpleMatrix, still need to create temporary (no in-place add for SimpleMatrix yet)
      temp_grad_w2 = @h.as(SimpleMatrix).transpose * d_out
      @g_w2 = @g_w2.as(SimpleMatrix) + temp_grad_w2

      # Efficient bias gradient accumulation for CPU path
      db2 = SimpleMatrix.zeros(1, d_out.cols)
      d_out.cols.times do |j|
        sum = 0.0
        d_out.rows.times { |i| sum += d_out[i, j] }
        db2[0, j] = sum
      end
      @g_b2 = @g_b2.as(SimpleMatrix) + db2

      drelu = relu_grad(@h.as(SimpleMatrix), dh)

      # For SimpleMatrix, still need to create temporary (no in-place add for SimpleMatrix yet)
      temp_grad_w1 = @x.not_nil!.as(SimpleMatrix).transpose * drelu
      @g_w1 = @g_w1.as(SimpleMatrix) + temp_grad_w1

      # Efficient bias gradient accumulation for CPU path
      db1 = SimpleMatrix.zeros(1, drelu.cols)
      drelu.cols.times do |j|
        sum = 0.0
        drelu.rows.times { |i| sum += drelu[i, j] }
        db1[0, j] = sum
      end
      @g_b1 = @g_b1.as(SimpleMatrix) + db1

      w1_cpu = @w1.as(SimpleMatrix)
      d_input = drelu * w1_cpu.transpose
      d_input
    end

    def apply_gradients(lr : Float64)
      # Check device type and call appropriate method
      if @w1.is_a?(CudaMatrix)
        apply_gradients_gpu(lr)
      else
        apply_gradients_cpu(lr)
      end
    end

    # GPU path gradient application - all CudaMatrix operations with in-place updates
    private def apply_gradients_gpu(lr : Float64)
      # Use in-place weight updates to eliminate matrix creation
      @w1.as(CudaMatrix).weight_update!(@g_w1.as(CudaMatrix), lr)
      @b1.as(CudaMatrix).weight_update!(@g_b1.as(CudaMatrix), lr)
      @w2.as(CudaMatrix).weight_update!(@g_w2.as(CudaMatrix), lr)
      @b2.as(CudaMatrix).weight_update!(@g_b2.as(CudaMatrix), lr)

      # Clear gradients in-place
      @g_w1.as(CudaMatrix).zero!
      @g_w2.as(CudaMatrix).zero!
      @g_b1.as(CudaMatrix).zero!
      @g_b2.as(CudaMatrix).zero!
    end

    # CPU path gradient application - all SimpleMatrix operations
    private def apply_gradients_cpu(lr : Float64)
      @w1 = @w1.as(SimpleMatrix) - @g_w1.as(SimpleMatrix) * lr
      @b1 = @b1.as(SimpleMatrix) - @g_b1.as(SimpleMatrix) * lr
      @w2 = @w2.as(SimpleMatrix) - @g_w2.as(SimpleMatrix) * lr
      @b2 = @b2.as(SimpleMatrix) - @g_b2.as(SimpleMatrix) * lr

      @g_w1 = SimpleMatrix.zeros(@w1.rows, @w1.cols)
      @g_w2 = SimpleMatrix.zeros(@w2.rows, @w2.cols)
      @g_b1 = SimpleMatrix.zeros(@b1.rows, @b1.cols)
      @g_b2 = SimpleMatrix.zeros(@b2.rows, @b2.cols)
    end

    def zero_gradients
      # Use in-place zeroing instead of creating new matrices
      if @w1.is_a?(CudaMatrix)
        @g_w1.as(CudaMatrix).zero!
        @g_w2.as(CudaMatrix).zero!
        @g_b1.as(CudaMatrix).zero!
        @g_b2.as(CudaMatrix).zero!
      else
        # CPU fallback - create new zero matrices for SimpleMatrix (no in-place zero yet)
        @g_w1 = SimpleMatrix.zeros(@w1.rows, @w1.cols)
        @g_w2 = SimpleMatrix.zeros(@w2.rows, @w2.cols)
        @g_b1 = SimpleMatrix.zeros(@b1.rows, @b1.cols)
        @g_b2 = SimpleMatrix.zeros(@b2.rows, @b2.cols)
      end
    end

    private def relu_grad(m : CudaMatrix, grad : CudaMatrix) : CudaMatrix
      # Use cuDNN for optimized ReLU gradient if available
      if CUDNN.available?
        begin
          result = CudaMatrix.new(grad.rows, grad.cols)
          CUDNN.relu_backward(m, grad, result)
          return result
        rescue e : Exception
          Log.debug { "cuDNN ReLU backward failed: #{e}, falling back to CUDA kernel" }
        end
      end

      # Use GPU kernel for ReLU gradient if available
      if CUDA.fully_available?
        begin
          result = grad.clone
          # Use CUDA kernel for ReLU backward pass
          CUDA.relu_backward(result.device_ptr.not_nil!, m.device_ptr.not_nil!, grad.device_ptr.not_nil!, m.rows * m.cols)
          result.mark_device_dirty!
          return result
        rescue e : Exception
          # Fall back to CPU computation if CUDA fails
        end
      end

      # CPU fallback - sync matrices to host first
      m.sync_from_device!("ff_gradient_debug")
      grad.sync_from_device!("ff_gradient_debug")
      result = grad.clone
      # Use unsafe_get for better performance
      m.rows.times do |i|
        m.cols.times do |j|
          result.unsafe_set(i, j, m.unsafe_get(i, j) > 0 ? grad.unsafe_get(i, j) : 0.0)
        end
      end
      result.sync_to_device!("ff_backward_result")
      result
    end

    private def relu_grad(m : SimpleMatrix, grad : SimpleMatrix) : SimpleMatrix
      out = grad.clone
      m.rows.times do |i|
        m.cols.times do |j|
          out[i, j] = m[i, j] > 0 ? grad[i, j] : 0.0
        end
      end
      out
    end

    # Optimized bias gradient accumulation with minimal CPU-GPU sync
    private def accumulate_bias_gradient(bias_grad : SimpleMatrix | CudaMatrix, d_out : CudaMatrix)
      if CUDA.fully_available? && bias_grad.is_a?(CudaMatrix)
        begin
          CUDA.row_sum(bias_grad.as(CudaMatrix).device_ptr.not_nil!, d_out.device_ptr.not_nil!, d_out.rows, d_out.cols)
          bias_grad.as(CudaMatrix).mark_device_dirty!
          return
        rescue e : Exception
          # Log GPU failure but continue with CPU fallback
          Log.debug { "GPU bias gradient accumulation failed: #{e.message}" }
        end
      end

      # CPU fallback - sync once and use batch operations
      d_out.sync_from_device!("ff_backward") if d_out.device_dirty?

      if bias_grad.is_a?(CudaMatrix)
        # Reuse existing workspace or create temporary accumulator to avoid repeated GPU syncs
        if @workspace_temp_bias.nil? || @workspace_temp_bias.not_nil!.cols != d_out.cols
          @workspace_temp_bias = CudaMatrix.zeros(1, d_out.cols)
        else
          @workspace_temp_bias.not_nil!.zero!
        end

        temp_bias = @workspace_temp_bias.not_nil!
        d_out.cols.times do |j|
          sum = 0.0
          d_out.rows.times { |i| sum += d_out.unsafe_get(i, j) }
          temp_bias.unsafe_set(0, j, sum)
        end
        temp_bias.sync_to_device!("ff_bias_update")
        bias_grad.as(CudaMatrix).add!(temp_bias)
      else
        # Direct accumulation for SimpleMatrix
        d_out.cols.times do |j|
          d_out.rows.times { |i| bias_grad[0, j] += d_out.unsafe_get(i, j) }
        end
      end
    end
  end
end
