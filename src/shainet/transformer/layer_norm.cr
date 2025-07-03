module SHAInet
  class LayerNorm
    getter gamma : SimpleMatrix | CudaMatrix
    getter beta : SimpleMatrix | CudaMatrix
    property g_gamma : SimpleMatrix | CudaMatrix
    property g_beta : SimpleMatrix | CudaMatrix

    # Allow setting gamma and beta for test compatibility
    def gamma=(val : SimpleMatrix | CudaMatrix)
      @gamma = val
    end

    def beta=(val : SimpleMatrix | CudaMatrix)
      @beta = val
    end

    @epsilon : Float64
    @x : SimpleMatrix | CudaMatrix | Nil
    @mean : SimpleMatrix | CudaMatrix
    @var : SimpleMatrix | CudaMatrix
    @norm : SimpleMatrix | CudaMatrix

    def initialize(d_model : Int32, epsilon : Float64 = 1e-5)
      # Use CudaMatrix when CUDA is available for better performance
      mat_klass = CUDA.fully_available? ? CudaMatrix : SimpleMatrix
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

    # Convert all internal matrices to GPU
    def to_gpu!
      if CUDA.fully_available?
        @gamma = @gamma.as(SimpleMatrix).to_cuda unless @gamma.is_a?(CudaMatrix)
        @beta = @beta.as(SimpleMatrix).to_cuda unless @beta.is_a?(CudaMatrix)
        @g_gamma = @g_gamma.as(SimpleMatrix).to_cuda unless @g_gamma.is_a?(CudaMatrix)
        @g_beta = @g_beta.as(SimpleMatrix).to_cuda unless @g_beta.is_a?(CudaMatrix)
        @mean = @mean.as(SimpleMatrix).to_cuda if @mean && !@mean.is_a?(CudaMatrix)
        @var = @var.as(SimpleMatrix).to_cuda if @var && !@var.is_a?(CudaMatrix)
        @norm = @norm.as(SimpleMatrix).to_cuda if @norm && !@norm.is_a?(CudaMatrix)
        @x = @x.as(SimpleMatrix).to_cuda if @x && !@x.is_a?(CudaMatrix)
      end
    end

    # GPU path - all CudaMatrix operations
    def forward(x : CudaMatrix) : CudaMatrix
      @x = x
      rows = x.rows
      cols = x.cols

      # Convert internal matrices to CUDA to match input
      cuda_mean = CudaMatrix.new(rows, 1)
      cuda_var = CudaMatrix.new(rows, 1)
      cuda_norm = CudaMatrix.new(rows, cols)

      begin
        # Try to use CUDA kernels - if they fail, fallback to CPU
        CUDA.row_mean_var(x.device_ptr.not_nil!,
          cuda_mean.device_ptr.not_nil!,
          cuda_var.device_ptr.not_nil!, rows, cols)
        CUDA.layer_norm(cuda_norm.device_ptr.not_nil!,
          x.device_ptr.not_nil!,
          cuda_mean.device_ptr.not_nil!,
          cuda_var.device_ptr.not_nil!,
          rows, cols, @epsilon)

        # Don't sync from device - keep data on GPU for performance
        # Only sync the intermediate results when actually needed for backward pass
        cuda_mean.mark_device_dirty!
        cuda_var.mark_device_dirty!
        cuda_norm.mark_device_dirty!

        # Store GPU references for backward pass instead of syncing to CPU
        @mean = cuda_mean # Keep as CudaMatrix
        @var = cuda_var   # Keep as CudaMatrix
        @norm = cuda_norm # Keep as CudaMatrix

        result = cuda_norm.clone
        # Apply gamma and beta - they should already be CudaMatrix in GPU path
        result.mul_row_vector!(@gamma.as(CudaMatrix))
        result.add_bias!(@beta.as(CudaMatrix))
        return result
      rescue e : Exception
        # Fall through to CPU implementation when kernels fail
      end

      # CPU fallback - compute mean and variance per row
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

      result = @norm.clone.as(CudaMatrix)
      result.mul_row_vector!(@gamma.as(CudaMatrix))
      result.add_bias!(@beta.as(CudaMatrix))
      result
    end

    # CPU path - all SimpleMatrix operations
    def forward(x : SimpleMatrix) : SimpleMatrix
      @x = x
      rows = x.rows
      cols = x.cols

      # CPU implementation - compute mean and variance per row
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
      # CPU operations: element-wise multiplication and addition
      result.rows.times do |i|
        result.cols.times do |j|
          result[i, j] = result[i, j] * @gamma.as(SimpleMatrix)[0, j] + @beta.as(SimpleMatrix)[0, j]
        end
      end
      result.as(SimpleMatrix)
    end

    # GPU path backward - all CudaMatrix operations
    def backward(d_out : CudaMatrix) : CudaMatrix
      x = @x.as(CudaMatrix)
      rows = x.rows
      cols = x.cols

      # If all tensors are CUDA and kernels are available, use GPU computation
      if CUDA.fully_available? && @mean.is_a?(CudaMatrix) && @var.is_a?(CudaMatrix) && @norm.is_a?(CudaMatrix)
        begin
          # Create GPU matrices for gradients
          d_x = CudaMatrix.new(rows, cols)
          d_gamma = CudaMatrix.zeros(1, cols)
          d_beta = CudaMatrix.zeros(1, cols)

          # Run CUDA kernel for backward pass
          CUDA.layer_norm_backward(
            d_x.device_ptr.not_nil!,
            d_gamma.device_ptr.not_nil!,
            d_beta.device_ptr.not_nil!,
            d_out.device_ptr.not_nil!,
            x.device_ptr.not_nil!,
            @gamma.as(CudaMatrix).device_ptr.not_nil!,
            @mean.as(CudaMatrix).device_ptr.not_nil!,
            @var.as(CudaMatrix).device_ptr.not_nil!,
            @norm.as(CudaMatrix).device_ptr.not_nil!,
            rows, cols, @epsilon
          )

          # Mark results as dirty on device
          d_x.mark_device_dirty!
          d_gamma.mark_device_dirty!
          d_beta.mark_device_dirty!

          # Only sync gradients when needed - accumulate gradients to CPU parameters
          # without calling sync_from_device! unnecessarily
          d_gamma.sync_from_device!
          d_beta.sync_from_device!

          # Accumulate gradients to parameters (always SimpleMatrix)
          cols.times do |j|
            @g_gamma[0, j] += d_gamma[0, j]
            @g_beta[0, j] += d_beta[0, j]
          end

          return d_x
        rescue e : Exception
          # Fall back to CPU implementation
        end
      end

      # CPU fallback - Only sync x if it's a CUDA matrix since we need CPU access
      x.sync_from_device!

      # Use CPU matrices for computation
      d_gamma = SimpleMatrix.zeros(1, cols)
      d_beta = SimpleMatrix.zeros(1, cols)
      d_x = SimpleMatrix.new(rows, cols)

      # Also sync other matrices if they're CUDA
      if @mean.is_a?(CudaMatrix)
        @mean.as(CudaMatrix).sync_from_device!
      end
      if @var.is_a?(CudaMatrix)
        @var.as(CudaMatrix).sync_from_device!
      end
      if @norm.is_a?(CudaMatrix)
        @norm.as(CudaMatrix).sync_from_device!
      end

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

      # Accumulate gradients to parameters
      cols.times do |j|
        @g_gamma[0, j] += d_gamma[0, j]
        @g_beta[0, j] += d_beta[0, j]
      end

      # Convert result back to CudaMatrix
      result = CudaMatrix.new(rows, cols)
      rows.times do |i|
        cols.times do |j|
          result[i, j] = d_x[i, j]
        end
      end
      result.sync_to_device!
      result
    end

    # CPU path backward - all SimpleMatrix operations
    def backward(d_out : SimpleMatrix) : SimpleMatrix
      x = @x.as(SimpleMatrix)
      rows = x.rows
      cols = x.cols

      # Use CPU matrices for computation
      d_gamma = SimpleMatrix.zeros(1, cols)
      d_beta = SimpleMatrix.zeros(1, cols)
      d_x = SimpleMatrix.new(rows, cols)

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

      # Accumulate gradients to parameters
      @g_gamma = @g_gamma.as(SimpleMatrix) + d_gamma
      @g_beta = @g_beta.as(SimpleMatrix) + d_beta
      d_x
    end

    def apply_gradients(lr : Float64)
      # Check device type and call appropriate method
      if @gamma.is_a?(CudaMatrix)
        apply_gradients_gpu(lr)
      else
        apply_gradients_cpu(lr)
      end
    end

    # GPU path gradient application - all CudaMatrix operations
    private def apply_gradients_gpu(lr : Float64)
      @gamma = @gamma.as(CudaMatrix) - @g_gamma.as(CudaMatrix) * lr
      @beta = @beta.as(CudaMatrix) - @g_beta.as(CudaMatrix) * lr
      @g_gamma = CudaMatrix.zeros(@gamma.rows, @gamma.cols)
      @g_beta = CudaMatrix.zeros(@beta.rows, @beta.cols)
    end

    # CPU path gradient application - all SimpleMatrix operations
    private def apply_gradients_cpu(lr : Float64)
      @gamma = @gamma.as(SimpleMatrix) - @g_gamma.as(SimpleMatrix) * lr
      @beta = @beta.as(SimpleMatrix) - @g_beta.as(SimpleMatrix) * lr
      @g_gamma = SimpleMatrix.zeros(@gamma.rows, @gamma.cols)
      @g_beta = SimpleMatrix.zeros(@beta.rows, @beta.cols)
    end

    def zero_gradients
      # Check device type and call appropriate method
      if @gamma.is_a?(CudaMatrix)
        @g_gamma = CudaMatrix.zeros(@gamma.rows, @gamma.cols)
        @g_beta = CudaMatrix.zeros(@beta.rows, @beta.cols)
      else
        @g_gamma = SimpleMatrix.zeros(@gamma.rows, @gamma.cols)
        @g_beta = SimpleMatrix.zeros(@beta.rows, @beta.cols)
      end
    end
  end
end
