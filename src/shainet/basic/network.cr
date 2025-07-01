require "./matrix_layer"
require "../math/unified_matrix"

module SHAInet
  class Network
    getter layers : Array(MatrixLayer)

    def initialize
      @layers = [] of MatrixLayer
    end

    def add_layer(in_size : Int32, out_size : Int32)
      layer = MatrixLayer.new(in_size, out_size)
      @layers << layer
      layer
    end

    def forward(input : UnifiedMatrix::MatrixData) : UnifiedMatrix::MatrixData
      result = input
      @layers.each do |layer|
        result = layer.forward(result)
      end
      result
    end

    def backward(grad : UnifiedMatrix::MatrixData) : UnifiedMatrix::MatrixData
      g = grad
      @layers.reverse_each do |layer|
        g = layer.backward(g)
      end
      g
    end
  end
end
