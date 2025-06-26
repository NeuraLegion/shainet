module SHAInet
  class PositionWiseFF
    getter w1, b1, w2, b2
    @w1 : SimpleMatrix
    @b1 : SimpleMatrix
    @w2 : SimpleMatrix
    @b2 : SimpleMatrix
    @g_w1 : SimpleMatrix
    @g_w2 : SimpleMatrix
    @g_b1 : SimpleMatrix
    @g_b2 : SimpleMatrix
    @x : SimpleMatrix?
    @h : SimpleMatrix
    @out : SimpleMatrix

    property g_w1 : SimpleMatrix
    property g_w2 : SimpleMatrix
    property g_b1 : SimpleMatrix
    property g_b2 : SimpleMatrix

    def initialize(d_model : Int32, hidden_dim : Int32)
      mat_klass = CUDA.available? ? CudaMatrix : SimpleMatrix
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

    def forward(x : SimpleMatrix)
      @x = x
      @h = x * @w1
      @h.rows.times do |i|
        @h.cols.times do |j|
          @h[i, j] += @b1[0, j]
        end
      end
      relu!(@h)
      @out = @h * @w2
      @out.rows.times do |i|
        @out.cols.times do |j|
          @out[i, j] += @b2[0, j]
        end
      end
      @out
    end

    def backward(d_out : SimpleMatrix)
      dh = d_out * @w2.transpose
      @g_w2 = @g_w2 + (@h.transpose * d_out)
      mat_klass = @w1.class
      db2 = mat_klass.zeros(1, d_out.cols)
      d_out.rows.times do |i|
        d_out.cols.times do |j|
          db2[0, j] += d_out[i, j]
        end
      end
      @g_b2 = @g_b2 + db2
      drelu = relu_grad(@h, dh)
      @g_w1 = @g_w1 + (@x.not_nil!.transpose * drelu)
      db1 = mat_klass.zeros(1, drelu.cols)
      drelu.rows.times do |i|
        drelu.cols.times do |j|
          db1[0, j] += drelu[i, j]
        end
      end
      @g_b1 = @g_b1 + db1
      d_input = drelu * @w1.transpose
      d_input
    end

    def apply_gradients(lr : Float64)
      mat_klass = @w1.class
      @w1 = @w1 - @g_w1 * lr
      @b1 = @b1 - @g_b1 * lr
      @w2 = @w2 - @g_w2 * lr
      @b2 = @b2 - @g_b2 * lr
      @g_w1 = mat_klass.zeros(@w1.rows, @w1.cols)
      @g_w2 = mat_klass.zeros(@w2.rows, @w2.cols)
      @g_b1 = mat_klass.zeros(@b1.rows, @b1.cols)
      @g_b2 = mat_klass.zeros(@b2.rows, @b2.cols)
    end

    def zero_gradients
      mat_klass = @w1.class
      @g_w1 = mat_klass.zeros(@w1.rows, @w1.cols)
      @g_w2 = mat_klass.zeros(@w2.rows, @w2.cols)
      @g_b1 = mat_klass.zeros(@b1.rows, @b1.cols)
      @g_b2 = mat_klass.zeros(@b2.rows, @b2.cols)
    end

    private def relu!(m : SimpleMatrix)
      m.rows.times do |i|
        m.cols.times do |j|
          m[i, j] = m[i, j] > 0 ? m[i, j] : 0.0
        end
      end
    end

    private def relu_grad(m : SimpleMatrix, grad : SimpleMatrix)
      out = grad.clone
      m.rows.times do |i|
        m.cols.times do |j|
          out[i, j] = m[i, j] > 0 ? grad[i, j] : 0.0
        end
      end
      out
    end
  end
end
