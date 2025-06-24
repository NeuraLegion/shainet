module SHAInet
  class LayerNorm
    getter gamma : SimpleMatrix
    getter beta : SimpleMatrix
    property g_gamma : SimpleMatrix
    property g_beta : SimpleMatrix

    @epsilon : Float64
    @x : SimpleMatrix?
    @mean : SimpleMatrix
    @var : SimpleMatrix
    @norm : SimpleMatrix

    def initialize(d_model : Int32, epsilon : Float64 = 1e-5)
      @gamma = SimpleMatrix.ones(1, d_model)
      @beta = SimpleMatrix.zeros(1, d_model)
      @g_gamma = SimpleMatrix.zeros(1, d_model)
      @g_beta = SimpleMatrix.zeros(1, d_model)
      @epsilon = epsilon
      @mean = SimpleMatrix.zeros(1, 1)
      @var = SimpleMatrix.zeros(1, 1)
      @norm = SimpleMatrix.zeros(1, 1)
    end

    def forward(x : SimpleMatrix)
      @x = x
      rows = x.rows
      cols = x.cols
      @mean = SimpleMatrix.new(rows, 1)
      @var = SimpleMatrix.new(rows, 1)
      @norm = SimpleMatrix.new(rows, cols)
      out = SimpleMatrix.new(rows, cols)
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
          out[i, j] = n * @gamma[0, j] + @beta[0, j]
        end
      end
      out
    end

    def backward(d_out : SimpleMatrix)
      x = @x.not_nil!
      rows = x.rows
      cols = x.cols
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
      @g_gamma = @g_gamma + d_gamma
      @g_beta = @g_beta + d_beta
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
