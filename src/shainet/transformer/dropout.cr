module SHAInet
  # Utility methods for dropout within Transformer layers
  module TransformerDropout
    # Returns a new SimpleMatrix where approximately `drop_percent` percent of
    # entries are set to 0.0. `drop_percent` should be between 0 and 100.
    def self.apply(matrix : SimpleMatrix, drop_percent : Int32)
      SHAInet.dropout(matrix, drop_percent)
    end

    def self.apply(matrix : CudaMatrix, drop_percent : Int32)
      SHAInet.dropout(matrix, drop_percent)
    end

    # Applies dropout to the given matrix in-place and returns it. `drop_percent`
    # should be between 0 and 100.
    def self.apply!(matrix : SimpleMatrix, drop_percent : Int32)
      raise ArgumentError.new("drop_percent must be between 0 and 100") unless 0 <= drop_percent && drop_percent <= 100
      matrix.dropout!(drop_percent.to_f / 100.0)
    end

    # Applies dropout to the given CUDA matrix in-place and returns it.
    def self.apply!(matrix : CudaMatrix, drop_percent : Int32)
      raise ArgumentError.new("drop_percent must be between 0 and 100") unless 0 <= drop_percent && drop_percent <= 100
      matrix.dropout!(drop_percent.to_f / 100.0)
    end
  end
end
