module SHAInet
  class LayerNorm
    getter gamma : SimpleMatrix
    getter beta : SimpleMatrix
    property g_gamma : SimpleMatrix
    property g_beta : SimpleMatrix

    # Allow setting gamma and beta for test compatibility
    def gamma=(val : SimpleMatrix)
      @gamma = val
    end

    def beta=(val : SimpleMatrix)
      @beta = val
    end

    @epsilon : Float64
    @x : SimpleMatrix?
    @mean : SimpleMatrix
    @var : SimpleMatrix
    @norm : SimpleMatrix

    def initialize(d_model : Int32, epsilon : Float64 = 1e-5)
      # Always use SimpleMatrix for compatibility - CUDA operations handled in forward/backward
      mat_klass = SimpleMatrix
      @gamma = mat_klass.new(1, d_model)
      d_model.times { |j| @gamma[0, j] = 1.0 }
      @beta = mat_klass.zeros(1, d_model)
      @g_gamma = mat_klass.zeros(1, d_model)
      @g_beta = mat_klass.zeros(1, d_model)
      @epsilon = epsilon
      @mean = mat_klass.zeros(1, 1)
      @var = mat_klass.zeros(1, 1)
      @norm = mat_klass.zeros(1, 1)
    end

    def forward(x : SimpleMatrix)
      @x = x
      rows = x.rows
      cols = x.cols

      # Convert to CUDA if input is CUDA and CUDA is available
      if CUDA.fully_available? && x.is_a?(CudaMatrix)
        # Convert internal matrices to CUDA to match input
        cuda_mean = CudaMatrix.new(rows, 1)
        cuda_var = CudaMatrix.new(rows, 1)
        cuda_norm = CudaMatrix.new(rows, cols)

        cx = x.as(CudaMatrix)
        begin
          # Try to use CUDA kernels - if they fail, fallback to CPU
          CUDA.row_mean_var(cuda_mean.device_ptr.not_nil!,
            cuda_var.device_ptr.not_nil!,
            cx.device_ptr.not_nil!, rows, cols)
          CUDA.layer_norm(cuda_norm.device_ptr.not_nil!,
            cx.device_ptr.not_nil!,
            cuda_mean.device_ptr.not_nil!,
            cuda_var.device_ptr.not_nil!,
            rows, cols, @epsilon)

          # Don't sync from device - keep data on GPU for performance
          # Only sync the intermediate results when actually needed for backward pass
          cuda_mean.mark_device_dirty!
          cuda_var.mark_device_dirty!
          cuda_norm.mark_device_dirty!

          # Store GPU references for backward pass instead of syncing to CPU
          @mean = cuda_mean  # Keep as CudaMatrix
          @var = cuda_var    # Keep as CudaMatrix
          @norm = cuda_norm  # Keep as CudaMatrix

          result = cuda_norm.clone
          # Apply gamma and beta - use their actual types for operations
          g = GPUMemory.keep_on_gpu(@gamma)
          b = GPUMemory.keep_on_gpu(@beta)
          result.mul_row_vector!(g.as(CudaMatrix))
          result.add_bias!(b.as(CudaMatrix))
          return result
        rescue e : Exception
          # Fall through to CPU implementation when kernels fail
        end
      end

      # CPU implementation or fallback - compute mean and variance per row
      @mean = SimpleMatrix.new(rows, 1)
      @var = SimpleMatrix.new(rows, 1)
      @norm = SimpleMatrix.new(rows, cols)
      rows.times do |i|
        mean = 0.0
        cols.times { |j| mean += x[i, j] }
        mean /= cols
        @mean[i, 0] = mean
        var = 0.0
        cols.times do |j|
          diff = x[i, j] - mean
          var += diff*diff
        end
        var /= cols
        @var[i, 0] = var
        denom = Math.sqrt(var + @epsilon)
        cols.times do |j|
          n = (x[i, j] - mean) / denom
          @norm[i, j] = n
        end
      end

      result = @norm.clone
      result.mul_row_vector!(@gamma)
      result.add_bias!(@beta)
      result
    end

    def backward(d_out : SimpleMatrix)
      x = @x.not_nil!
      x.sync_from_device! if x.responds_to?(:sync_from_device!)
      rows = x.rows
      cols = x.cols

      # Use the same matrix type as the input for consistency
      mat_klass = d_out.is_a?(CudaMatrix) ? CudaMatrix : SimpleMatrix
      d_gamma = mat_klass.zeros(1, cols)
      d_beta = mat_klass.zeros(1, cols)
      d_x = mat_klass.new(rows, cols)
      rows.times do |i|
        denom = Math.sqrt(@var[i, 0] + @epsilon)
        inv = 1.0 / denom
        col_f = cols.to_f64
        sum_dout_gamma = 0.0
        sum_dout_gamma_norm = 0.0
        cols.times do |j|
          doutg = d_out[i, j] * @gamma[0, j]
          sum_dout_gamma += doutg
          sum_dout_gamma_norm += doutg * (x[i, j] - @mean[i, 0])
          d_gamma[0, j] += d_out[i, j] * @norm[i, j]
          d_beta[0, j] += d_out[i, j]
        end
        cols.times do |j|
          xm = x[i, j] - @mean[i, 0]
          doutg = d_out[i, j] * @gamma[0, j]
          d_x[i, j] = inv * (doutg - sum_dout_gamma/col_f - xm * inv*inv / col_f * sum_dout_gamma_norm)
        end
      end
      # Convert gradients to match parameter types if needed
      if d_gamma.is_a?(CudaMatrix)
        @g_gamma = @g_gamma + SimpleMatrix.from_a(d_gamma.to_a)
        @g_beta = @g_beta + SimpleMatrix.from_a(d_beta.to_a)
      else
        @g_gamma = @g_gamma + d_gamma
        @g_beta = @g_beta + d_beta
      end
      d_x
    end

    def apply_gradients(lr : Float64)
      @gamma = @gamma - @g_gamma * lr
      @beta = @beta - @g_beta * lr
      @g_gamma = SimpleMatrix.zeros(@gamma.rows, @gamma.cols)
      @g_beta = SimpleMatrix.zeros(@beta.rows, @beta.cols)
    end

    def zero_gradients
      @g_gamma = SimpleMatrix.zeros(@gamma.rows, @gamma.cols)
      @g_beta = SimpleMatrix.zeros(@beta.rows, @beta.cols)
    end
  end
end
