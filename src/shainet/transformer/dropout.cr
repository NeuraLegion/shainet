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
  end
end
