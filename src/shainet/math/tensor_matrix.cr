module SHAInet
  class TensorMatrix
    property rows : Int32
    property cols : Int32
    getter data : Array(Autograd::Tensor)

    def initialize(rows : Int32, cols : Int32, init : Autograd::Tensor = Autograd::Tensor.new(0.0))
      @rows = rows
      @cols = cols
      @data = Array(Autograd::Tensor).new(@rows * @cols, init)
    end

    def self.zeros(rows : Int32, cols : Int32)
      new(rows, cols, Autograd::Tensor.new(0.0))
    end

    def self.ones(rows : Int32, cols : Int32)
      new(rows, cols, Autograd::Tensor.new(1.0))
    end

    def []=(r : Int32, c : Int32, v : Autograd::Tensor)
      @data[r * @cols + c] = v
    end

    def [](r : Int32, c : Int32)
      @data[r * @cols + c]
    end

    def +(other : TensorMatrix)
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      result = TensorMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = self[i, j] + other[i, j]
        end
      end
      result
    end

    def -(other : TensorMatrix)
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      result = TensorMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = self[i, j] - other[i, j]
        end
      end
      result
    end

    def *(other : TensorMatrix)
      raise ArgumentError.new("size mismatch") unless @cols == other.rows
      result = TensorMatrix.new(@rows, other.cols)
      @rows.times do |i|
        other.cols.times do |j|
          sum = Autograd::Tensor.new(0.0)
          @cols.times do |k|
            sum = sum + self[i, k] * other[k, j]
          end
          result[i, j] = sum
        end
      end
      result
    end

    def *(scalar : Number)
      result = TensorMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = self[i, j] * scalar
        end
      end
      result
    end

    def transpose
      result = TensorMatrix.new(@cols, @rows)
      @rows.times do |i|
        @cols.times do |j|
          result[j, i] = self[i, j]
        end
      end
      result
    end

    def self.from_a(array : Array(Array(Float64)))
      rows = array.size
      cols = array.first.size
      m = TensorMatrix.new(rows, cols)
      rows.times do |i|
        cols.times do |j|
          m[i, j] = Autograd::Tensor.new(array[i][j])
        end
      end
      m
    end

    def random_fill!(min : Float64 = -0.1, max : Float64 = 0.1)
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] = Autograd::Tensor.new(rand(min..max))
        end
      end
      self
    end

    def slice_cols(start_col : Int32, length : Int32)
      result = TensorMatrix.new(@rows, length)
      @rows.times do |i|
        length.times do |j|
          result[i, j] = self[i, start_col + j]
        end
      end
      result
    end

    def set_cols!(start_col : Int32, other : TensorMatrix)
      raise ArgumentError.new("row mismatch") unless other.rows == @rows
      other.cols.times do |j|
        @rows.times do |i|
          self[i, start_col + j] = other[i, j]
        end
      end
    end

    def clone
      dup = TensorMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          dup[i, j] = self[i, j]
        end
      end
      dup
    end

    def to_simple
      m = SimpleMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          m[i, j] = self[i, j].data
        end
      end
      m
    end

    def zero_grads!
      @data.each(&.grad=(0.0))
    end
  end
end
