module SHAInet
  # RMSNorm: Root Mean Square Layer Normalization (used in LLaMA/Mistral)
  # Unlike LayerNorm, RMSNorm does not subtract the mean and has no beta (bias).
  # Formula: output = x / sqrt(mean(x^2) + eps) * gamma
  class RMSNorm
    property gamma : SimpleMatrix | CudaMatrix
    getter size : Int32
    @eps : Float64

    def initialize(@size : Int32, @eps : Float64 = 1e-6)
      @gamma = SimpleMatrix.new(1, @size, 1.0)
    end

    def forward(x : SimpleMatrix) : SimpleMatrix
      rows = x.rows
      cols = x.cols
      result = SimpleMatrix.new(rows, cols)

      rows.times do |i|
        # Compute RMS for this row
        sq_sum = 0.0
        cols.times { |j| v = x[i, j]; sq_sum += v * v }
        rms = Math.sqrt(sq_sum / cols + @eps)

        # Normalize and scale by gamma
        cols.times do |j|
          result[i, j] = (x[i, j] / rms) * @gamma.as(SimpleMatrix)[0, j]
        end
      end

      result
    end
  end
end
