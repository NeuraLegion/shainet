module SHAInet
  class SimpleMatrix
    property rows : Int32
    property cols : Int32
    getter data : Array(Float64)

    def initialize(@rows : Int32, @cols : Int32, init : Float64 = 0.0)
      @data = Array(Float64).new(@rows * @cols, init)
    end

    def self.zeros(rows : Int32, cols : Int32)
      new(rows, cols, 0.0)
    end

    def self.ones(rows : Int32, cols : Int32)
      new(rows, cols, 1.0)
    end

    def self.tensor(rows : Int32, cols : Int32)
      TensorMatrix.new(rows, cols)
    end

    def [](r : Int32, c : Int32)
      @data[r * @cols + c]
    end

    def []=(r : Int32, c : Int32, v : Float64)
      @data[r * @cols + c] = v
    end

    def +(other : SimpleMatrix)
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      result = SimpleMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = self[i, j] + other[i, j]
        end
      end
      result
    end

    def -(other : SimpleMatrix)
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      result = SimpleMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = self[i, j] - other[i, j]
        end
      end
      result
    end

    def *(other : SimpleMatrix)
      raise ArgumentError.new("size mismatch") unless @cols == other.rows
      result = SimpleMatrix.new(@rows, other.cols)
      @rows.times do |i|
        other.cols.times do |j|
          sum = 0.0
          @cols.times do |k|
            sum += self[i, k] * other[k, j]
          end
          result[i, j] = sum
        end
      end
      result
    end

    def *(scalar : Number)
      result = SimpleMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = self[i, j] * scalar.to_f64
        end
      end
      result
    end

    def transpose
      result = SimpleMatrix.new(@cols, @rows)
      @rows.times do |i|
        @cols.times do |j|
          result[j, i] = self[i, j]
        end
      end
      result
    end

    # Transpose the matrix into an existing destination matrix in-place.
    # This avoids allocating a new matrix on each call.
    def transpose_into!(dest : SimpleMatrix)
      raise ArgumentError.new("size mismatch") unless dest.rows == @cols && dest.cols == @rows

      @rows.times do |i|
        @cols.times do |j|
          dest[j, i] = self[i, j]
        end
      end

      dest
    end

    def to_a
      Array.new(@rows) do |i|
        Array.new(@cols) do |j|
          self[i, j]
        end
      end
    end

    # Construct a matrix from a nested Array
    def self.from_a(array : Array(Array(GenNum)))
      rows = array.size
      cols = array.first.size
      m = SimpleMatrix.new(rows, cols)
      rows.times do |i|
        cols.times do |j|
          m[i, j] = array[i][j].to_f64
        end
      end
      m
    end

    # Fill the matrix with random values in the given range
    def random_fill!(min : Float64 = -0.1, max : Float64 = 0.1)
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] = rand(min..max)
        end
      end
      self
    end

    # Slice a range of columns into the provided destination matrix.
    def slice_cols_into!(dest : SimpleMatrix, start_col : Int32, length : Int32)
      raise ArgumentError.new("size mismatch") unless dest.rows == @rows && dest.cols == length
      @rows.times do |i|
        length.times do |j|
          dest[i, j] = self[i, start_col + j]
        end
      end
      dest
    end

    # Slice a range of columns from the matrix
    def slice_cols(start_col : Int32, length : Int32)
      result = SimpleMatrix.new(@rows, length)
      slice_cols_into!(result, start_col, length)
      result
    end

    # Set a range of columns in-place from another matrix
    def set_cols!(start_col : Int32, other : SimpleMatrix)
      raise ArgumentError.new("row mismatch") unless other.rows == @rows
      other.cols.times do |j|
        @rows.times do |i|
          self[i, start_col + j] = other[i, j]
        end
      end
    end

    def clone
      dup = SimpleMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          dup[i, j] = self[i, j]
        end
      end
      dup
    end

    # In-place element-wise addition.
    def add!(other : SimpleMatrix)
      raise ArgumentError.new("size mismatch") unless other.rows == @rows && other.cols == @cols
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] += other[i, j]
        end
      end
      self
    end

    # Add a bias row vector to each row of the matrix in-place.
    def add_bias!(bias : SimpleMatrix)
      raise ArgumentError.new("bias size mismatch") unless bias.rows == 1 && bias.cols == @cols
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] += bias[0, j]
        end
      end
      self
    end

    # Element-wise ReLU activation in-place.
    def relu!
      @rows.times do |i|
        @cols.times do |j|
          v = self[i, j]
          self[i, j] = v > 0 ? v : 0.0
        end
      end
      self
    end

    # Apply dropout in-place using the given probability in the range 0.0..1.0.
    def dropout!(prob : Float64)
      raise ArgumentError.new("prob must be between 0 and 1") unless 0.0 <= prob && prob <= 1.0

      @rows.times do |i|
        @cols.times do |j|
          self[i, j] = Random.rand < prob ? 0.0 : self[i, j]
        end
      end

      self
    end

    # Multiply each column by the corresponding value in a row vector in-place.
    def mul_row_vector!(vec : SimpleMatrix)
      raise ArgumentError.new("vector size mismatch") unless vec.rows == 1 && vec.cols == @cols
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] *= vec[0, j]
        end
      end
      self
    end

    # Convert SimpleMatrix to CudaMatrix for GPU operations
    def to_cuda : CudaMatrix
      result = CudaMatrix.new(@rows, @cols)
      # Use batch copy through raw data for better performance
      @rows.times do |i|
        @cols.times do |j|
          result.unsafe_set(i, j, self[i, j])
        end
      end
      result.sync_to_device!("simple_to_cuda_conversion")
      result
    end

    # Apply softmax to each row in-place.
    def softmax_rows!
      @rows.times do |i|
        row_max = -Float64::INFINITY
        @cols.times { |j| row_max = Math.max(row_max, self[i, j]) }

        row_sum = 0.0
        @cols.times do |j|
          val = Math.exp(self[i, j] - row_max)
          self[i, j] = val
          row_sum += val
        end

        @cols.times do |j|
          self[i, j] = self[i, j] / row_sum
        end
      end
      self
    end
  end
end
