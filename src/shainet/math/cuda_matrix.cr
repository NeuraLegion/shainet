require "./simple_matrix"
require "../cuda"

module SHAInet
  # Basic GPU matrix wrapper. If CUDA is not available the operations
  # fall back to SimpleMatrix. This is only a minimal placeholder for
  # real GPU implementations.
  class CudaMatrix < SimpleMatrix
    def self.from_a(array : Array(Array(GenNum)))
      m = new(array.size, array.first.size)
      array.each_with_index do |row, i|
        row.each_with_index do |val, j|
          m[i, j] = val.to_f64
        end
      end
      m
    end

    def *(other : CudaMatrix)
      # Real GPU math should be implemented here. Currently uses CPU
      # implementation if CUDA is not available.
      super(other)
    end
  end
end
