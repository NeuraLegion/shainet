module SHAInet
  # RMSNorm: Root Mean Square Layer Normalization (used in LLaMA/Mistral)
  # Formula: output = x / sqrt(mean(x^2) + eps) * gamma
  class RMSNorm
    property gamma : SimpleMatrix | CudaMatrix
    getter size : Int32
    @eps : Float64

    def initialize(@size : Int32, @eps : Float64 = 1e-6)
      @gamma = SimpleMatrix.new(1, @size, 1.0)
    end

    def to_gpu!
      return unless CUDA.fully_available?
      @gamma = @gamma.as(SimpleMatrix).to_cuda unless @gamma.is_a?(CudaMatrix)
    end

    def forward(x : SimpleMatrix) : SimpleMatrix
      rows = x.rows
      cols = x.cols
      result = SimpleMatrix.new(rows, cols)

      # Read gamma values (might be on GPU)
      g = @gamma
      if g.is_a?(CudaMatrix)
        g.sync_from_device!("rmsnorm_gamma") if g.device_dirty?
      end

      rows.times do |i|
        sq_sum = 0.0
        cols.times { |j| v = x[i, j]; sq_sum += v * v }
        rms = Math.sqrt(sq_sum / cols + @eps)
        cols.times { |j| result[i, j] = (x[i, j] / rms) * g[0, j] }
      end

      result
    end

    # True when the norm can run entirely on the device: gamma already there and the
    # kernel present in the loaded .so.
    def device_capable? : Bool
      @gamma.is_a?(CudaMatrix) && CUDA.fully_available? && CUDA.block_device_kernels_available?
    end

    # Normalise a device-resident matrix into a caller-owned device destination.
    # Allocates nothing and never touches the host, so it can sit inside a
    # device-resident block chain.
    def forward_into(x : CudaMatrix, dst : CudaMatrix) : CudaMatrix
      g = @gamma.as(CudaMatrix)
      g.sync_to_device!("rmsnorm_gamma_up") unless g.device_dirty?
      x.sync_to_device!("rmsnorm_in") unless x.device_dirty?
      CUDA.rms_norm_forward(dst.device_ptr.not_nil!, x.device_ptr.not_nil!,
        g.device_ptr.not_nil!, x.rows, x.cols, @eps.to_f32)
      dst.mark_device_dirty!
      dst
    end

    def forward(x : CudaMatrix) : CudaMatrix
      # With the kernel present this is a pure device op. Without it, fall back to the
      # historical path below, which reads the row back, normalises on the host and
      # pushes it again.
      if device_capable?
        return forward_into(x, CudaMatrix.new(x.rows, x.cols))
      end

      x.sync_from_device!("rmsnorm") if x.device_dirty?
      gamma_sm = @gamma
      if gamma_sm.is_a?(CudaMatrix)
        gamma_sm.sync_from_device!("rmsnorm_gamma") if gamma_sm.device_dirty?
      end

      rows = x.rows
      cols = x.cols
      result = CudaMatrix.new(rows, cols)

      rows.times do |i|
        sq_sum = 0.0
        cols.times { |j| v = x[i, j]; sq_sum += v * v }
        rms = Math.sqrt(sq_sum / cols + @eps)
        cols.times do |j|
          g = gamma_sm.is_a?(CudaMatrix) ? gamma_sm[0, j] : gamma_sm.as(SimpleMatrix)[0, j]
          result[i, j] = (x[i, j] / rms) * g
        end
      end

      result.sync_to_device!("rmsnorm_done")
      result
    end
  end
end
