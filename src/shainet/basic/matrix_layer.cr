require "../math/simple_matrix"
require "../math/cuda_matrix"
require "../math/unified_matrix"

module SHAInet
  class MatrixLayer < UnifiedMatrix
    property weights : SimpleMatrix | CudaMatrix
    property biases : SimpleMatrix | CudaMatrix
    property g_w : SimpleMatrix | CudaMatrix
    property g_b : SimpleMatrix | CudaMatrix

    def initialize(in_size : Int32, out_size : Int32)
      mat_klass = CUDA.available? ? CudaMatrix : SimpleMatrix
      @weights = mat_klass.new(in_size, out_size).random_fill!
      @biases = mat_klass.new(1, out_size).random_fill!
      @g_w = mat_klass.zeros(in_size, out_size)
      @g_b = mat_klass.zeros(1, out_size)
      @input = nil
    end

    # Dummy implementations for abstract methods on UnifiedMatrix
    def forward(input : UnifiedMatrix) : UnifiedMatrix
      raise "Unsupported type"
    end

    def backward(grad : UnifiedMatrix) : UnifiedMatrix
      raise "Unsupported type"
    end

    def forward(input : SimpleMatrix) : SimpleMatrix
      @input = input
      out = input * @weights.as(SimpleMatrix)
      out.add_bias!(@biases.as(SimpleMatrix))
      out
    end

    def forward(input : SimpleMatrix | CudaMatrix)
      if input.is_a?(CudaMatrix)
        forward(input.as(CudaMatrix))
      else
        forward(input.as(SimpleMatrix))
      end
    end

    def forward(input : CudaMatrix) : SimpleMatrix | CudaMatrix
      @input = input
      w = @weights.as(CudaMatrix)
      b = @biases.as(CudaMatrix)
      out = input * w
      out.add_bias!(b)
      out
    end

    def backward(grad : SimpleMatrix) : SimpleMatrix
      input = @input.as(SimpleMatrix)
      @g_w = @g_w + input.transpose * grad
      grad.rows.times do |i|
        grad.cols.times do |j|
          @g_b[0, j] += grad[i, j]
        end
      end
      grad * @weights.as(SimpleMatrix).transpose
    end

    def backward(grad : CudaMatrix) : SimpleMatrix | CudaMatrix
      input = @input.as(CudaMatrix)
      @g_w = @g_w + input.transpose * grad
      if CUDA.available? && CUDA.kernels_available? && @g_b.is_a?(CudaMatrix)
        begin
          CUDA.row_sum(@g_b.as(CudaMatrix).device_ptr.not_nil!, grad.device_ptr.not_nil!, grad.rows, grad.cols)
          GPUMemory.batch_sync_from_device([@g_b])
        rescue
          grad.rows.times do |i|
            grad.cols.times do |j|
              @g_b[0, j] += grad[i, j]
            end
          end
        end
      else
        grad.rows.times do |i|
          grad.cols.times do |j|
            @g_b[0, j] += grad[i, j]
          end
        end
      end
      grad * @weights.as(CudaMatrix).transpose
    end

    def update_weights(lr : Float64)
      if @weights.is_a?(CudaMatrix)
        w = @weights.as(CudaMatrix)
        gw = @g_w.is_a?(CudaMatrix) ? @g_w.as(CudaMatrix) : SHAInet::GPUMemory.to_gpu(@g_w.as(SimpleMatrix))
        b = @biases.as(CudaMatrix)
        gb = @g_b.is_a?(CudaMatrix) ? @g_b.as(CudaMatrix) : SHAInet::GPUMemory.to_gpu(@g_b.as(SimpleMatrix))
        @weights = w - gw * lr
        @biases = b - gb * lr
        @g_w = CudaMatrix.zeros(gw.rows, gw.cols)
        @g_b = CudaMatrix.zeros(gb.rows, gb.cols)
      else
        w = @weights.as(SimpleMatrix)
        gw = @g_w.is_a?(SimpleMatrix) ? @g_w.as(SimpleMatrix) : SHAInet::SimpleMatrix.from_a(@g_w.as(CudaMatrix).to_a)
        b = @biases.as(SimpleMatrix)
        gb = @g_b.is_a?(SimpleMatrix) ? @g_b.as(SimpleMatrix) : SHAInet::SimpleMatrix.from_a(@g_b.as(CudaMatrix).to_a)
        @weights = w - gw * lr
        @biases = b - gb * lr
        @g_w = SimpleMatrix.zeros(gw.rows, gw.cols)
        @g_b = SimpleMatrix.zeros(gb.rows, gb.cols)
      end
    end

    @input : SimpleMatrix | CudaMatrix?
  end
end
