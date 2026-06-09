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

      rows.times do |i|
        sq_sum = 0.0
        cols.times { |j| v = x[i, j]; sq_sum += v * v }
        rms = Math.sqrt(sq_sum / cols + @eps)
        cols.times { |j| result[i, j] = (x[i, j] / rms) * @gamma.as(SimpleMatrix)[0, j] }
      end

      result
    end

    def forward(x : CudaMatrix) : CudaMatrix
      # RMSNorm on CPU then sync — small per-row operation
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
