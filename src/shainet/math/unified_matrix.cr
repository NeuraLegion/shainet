require "./simple_matrix"
require "./cuda_matrix"

module SHAInet
  abstract class UnifiedMatrix
    abstract def forward(input : UnifiedMatrix) : UnifiedMatrix
    abstract def backward(grad : UnifiedMatrix) : UnifiedMatrix
    abstract def update_weights(lr : Float64)

    alias MatrixData = SimpleMatrix | CudaMatrix

    # Stack matrices vertically into a single matrix. The returned
    # matrix keeps the same device type (CPU/GPU) as the inputs.
    def self.stack(matrices : Array(MatrixData)) : MatrixData
      raise ArgumentError.new("no matrices to stack") if matrices.empty?

      cols = matrices.first.cols
      total_rows = 0
      matrices.each do |m|
        raise ArgumentError.new("column mismatch") unless m.cols == cols
        total_rows += m.rows
      end

      result = matrices.first.is_a?(CudaMatrix) ? CudaMatrix.new(total_rows, cols) : SimpleMatrix.new(total_rows, cols)

      offset = 0
      matrices.each do |m|
        m.rows.times do |i|
          cols.times do |j|
            result[offset + i, j] = m[i, j]
          end
        end
        offset += m.rows
      end

      result
    end

    # Split a stacked matrix back into an array of matrices using the
    # provided row counts.
    def self.unstack(matrix : MatrixData, rows : Array(Int32)) : Array(MatrixData)
      parts = [] of MatrixData
      offset = 0
      rows.each do |r|
        part = matrix.is_a?(CudaMatrix) ? CudaMatrix.new(r, matrix.cols) : SimpleMatrix.new(r, matrix.cols)
        r.times do |i|
          matrix.cols.times do |j|
            part[i, j] = matrix[offset + i, j]
          end
        end
        offset += r
        parts << part
      end
      parts
    end
  end
end
