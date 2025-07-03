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

    # GPU path - all CudaMatrix operations
    def forward(x : CudaMatrix) : CudaMatrix
      @x = x
      # Weights are already CudaMatrix in GPU path
      w1_gpu = @w1.as(CudaMatrix)
      b1_gpu = @b1.as(CudaMatrix)
      w2_gpu = @w2.as(CudaMatrix)
      b2_gpu = @b2.as(CudaMatrix)

      @h = x * w1_gpu
      @h.as(CudaMatrix).add_bias!(b1_gpu)
      @h.as(CudaMatrix).relu!
      @out = @h.as(CudaMatrix) * w2_gpu
      @out.as(CudaMatrix).add_bias!(b2_gpu)
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
      dh = d_out * w2_gpu.transpose
      @g_w2 = @g_w2.as(CudaMatrix) + (@h.as(CudaMatrix).transpose * d_out)

      # Efficient bias gradient using GPU
      if CUDA.fully_available? && @g_b2.is_a?(CudaMatrix)
        begin
          CUDA.row_sum(@g_b2.as(CudaMatrix).device_ptr.not_nil!, d_out.device_ptr.not_nil!, d_out.rows, d_out.cols)
          @g_b2.as(CudaMatrix).mark_device_dirty!
        rescue e : Exception
          # Fallback to CPU computation
          d_out.rows.times do |i|
            d_out.cols.times do |j|
              @g_b2[0, j] += d_out[i, j]
            end
          end
        end
      else
        db2 = CudaMatrix.zeros(1, d_out.cols)
        d_out.rows.times do |i|
          d_out.cols.times do |j|
            db2[0, j] += d_out[i, j]
          end
        end
        @g_b2 = @g_b2.as(CudaMatrix) + db2
      end

      drelu = relu_grad(@h.as(CudaMatrix), dh)
      @g_w1 = @g_w1.as(CudaMatrix) + (@x.not_nil!.as(CudaMatrix).transpose * drelu)

      if CUDA.fully_available? && @g_b1.is_a?(CudaMatrix)
        begin
          CUDA.row_sum(@g_b1.as(CudaMatrix).device_ptr.not_nil!, drelu.device_ptr.not_nil!, drelu.rows, drelu.cols)
          @g_b1.as(CudaMatrix).mark_device_dirty!
        rescue e : Exception
          # Fallback to CPU computation
          drelu.rows.times do |i|
            drelu.cols.times do |j|
              @g_b1[0, j] += drelu[i, j]
            end
          end
        end
      else
        db1 = CudaMatrix.zeros(1, drelu.cols)
        drelu.rows.times do |i|
          drelu.cols.times do |j|
            db1[0, j] += drelu[i, j]
          end
        end
        @g_b1 = @g_b1.as(CudaMatrix) + db1
      end

      w1_gpu = @w1.as(CudaMatrix)
      d_input = drelu * w1_gpu.transpose
      d_input
    end

    # CPU path backward
    def backward(d_out : SimpleMatrix) : SimpleMatrix
      w2_cpu = @w2.as(SimpleMatrix)
      dh = d_out * w2_cpu.transpose
      @g_w2 = @g_w2.as(SimpleMatrix) + (@h.as(SimpleMatrix).transpose * d_out)

      db2 = SimpleMatrix.zeros(1, d_out.cols)
      d_out.rows.times do |i|
        d_out.cols.times do |j|
          db2[0, j] += d_out[i, j]
        end
      end
      @g_b2 = @g_b2.as(SimpleMatrix) + db2

      drelu = relu_grad(@h.as(SimpleMatrix), dh)
      @g_w1 = @g_w1.as(SimpleMatrix) + (@x.not_nil!.as(SimpleMatrix).transpose * drelu)

      db1 = SimpleMatrix.zeros(1, drelu.cols)
      drelu.rows.times do |i|
        drelu.cols.times do |j|
          db1[0, j] += drelu[i, j]
        end
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

    # GPU path gradient application - all CudaMatrix operations
    private def apply_gradients_gpu(lr : Float64)
      @w1 = @w1.as(CudaMatrix) - @g_w1.as(CudaMatrix) * lr
      @b1 = @b1.as(CudaMatrix) - @g_b1.as(CudaMatrix) * lr
      @w2 = @w2.as(CudaMatrix) - @g_w2.as(CudaMatrix) * lr
      @b2 = @b2.as(CudaMatrix) - @g_b2.as(CudaMatrix) * lr

      # Sync updated weights to device
      [@w1, @b1, @w2, @b2].each do |mat|
        mat.as(CudaMatrix).sync_to_device! unless mat.as(CudaMatrix).device_dirty?
      end

      @g_w1 = CudaMatrix.zeros(@w1.rows, @w1.cols)
      @g_w2 = CudaMatrix.zeros(@w2.rows, @w2.cols)
      @g_b1 = CudaMatrix.zeros(@b1.rows, @b1.cols)
      @g_b2 = CudaMatrix.zeros(@b2.rows, @b2.cols)
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
      # Check device type and use appropriate matrix type
      if @w1.is_a?(CudaMatrix)
        @g_w1 = CudaMatrix.zeros(@w1.rows, @w1.cols)
        @g_w2 = CudaMatrix.zeros(@w2.rows, @w2.cols)
        @g_b1 = CudaMatrix.zeros(@b1.rows, @b1.cols)
        @g_b2 = CudaMatrix.zeros(@b2.rows, @b2.cols)
      else
        @g_w1 = SimpleMatrix.zeros(@w1.rows, @w1.cols)
        @g_w2 = SimpleMatrix.zeros(@w2.rows, @w2.cols)
        @g_b1 = SimpleMatrix.zeros(@b1.rows, @b1.cols)
        @g_b2 = SimpleMatrix.zeros(@b2.rows, @b2.cols)
      end
    end

    private def relu_grad(m : CudaMatrix, grad : CudaMatrix) : CudaMatrix
      out = grad.clone
      m.rows.times do |i|
        m.cols.times do |j|
          out[i, j] = m[i, j] > 0 ? grad[i, j] : 0.0
        end
      end
      out
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
  end
end
