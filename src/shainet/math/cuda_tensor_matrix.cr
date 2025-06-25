require "./tensor_matrix"

module SHAInet
  class CudaTensorMatrix < TensorMatrix
    def initialize(rows : Int32, cols : Int32, init : Autograd::Tensor = Autograd::Tensor.new(0.0))
      super(rows, cols, init)
    end
  end
end
