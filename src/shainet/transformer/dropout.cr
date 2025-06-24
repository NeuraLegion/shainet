module SHAInet
  # Utility methods for dropout within Transformer layers
  module TransformerDropout
    # Returns a new SimpleMatrix where approximately `drop_percent` percent of
    # entries are set to 0.0. `drop_percent` should be between 0 and 100.
    def self.apply(matrix : SimpleMatrix, drop_percent : Int32)
      raise ArgumentError.new("drop_percent must be between 0 and 100") unless 0 <= drop_percent <= 100
      out = SimpleMatrix.new(matrix.rows, matrix.cols)
      matrix.rows.times do |i|
        matrix.cols.times do |j|
          out[i, j] = rand(0...100) < drop_percent ? 0.0 : matrix[i, j]
        end
      end
      out
    end
  end
end
