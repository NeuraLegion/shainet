require "logger"

module SHAInet
  # module PtrMatrix
  class PtrMatrix
    property data : Slice(Slice(Pointer(Float64)))

    def initialize(width : Int32, height : Int32)
      _temp = 0.0
      temp = pointerof(_temp)
      @data = Slice.new(height) { |row| row = Slice.new(width) { temp } }
    end

    def +(mat : PtrMatrix)
      raise MathError.new("Matrix dimention error (mat1_shape: #{self.shape}, mat2_shape: #{mat.shape})") if self.shape != mat.shape
      @data.size.times do |row|
        @data[row].size.times do |col|
          @data[row][col].value += mat.data[row][col].value
        end
      end
    end

    def elem_mult(mat : PtrMatrix)
      raise MathError.new("Matrix dimention error (mat1_shape: #{self.shape}, mat2_shape: #{mat.shape})") if self.shape != mat.shape
      @data.size.times do |row|
        @data[row].size.times do |col|
          @data[row][col].value *= mat.data[row][col].value
        end
      end
    end

    def static_dot(mat : PtrMatrix, mat_out : PtrMatrix)
      if self.shape[1] != mat.shape[0]
        raise MathError.new("Matrix dimention error, mat1 rows must equal mat2 columns")
      end
      mat.transpose

      @data.size.times do |v1|
        mat.data.size.times do |v2|
          mult_sum = 0_f64
          mat.data[v2].size.times do |i|
            mult_sum += (@data[v1][i].value * mat.data[v2][i].value)
          end
          mat_out.data[v1][v2].value = mult_sum
        end
      end
      mat.transpose
    end

    # def *(mat : PtrMatrix)
    #   raise MathError.new("Matrix dimention error (mat1_shape: #{self.shape}, mat2_shape: #{mat.shape})") if self.shape != mat.shape
    #   @data.size.times do |row|
    #     @data[row].size.times do |col|
    #       @data[row][col].value += mat.data[row][col].value
    #     end
    #   end
    # end

    def transpose
      temp = 0.0
      new_mat = PtrMatrix.new(@data.size, @data.first.size)

      @data.size.times do |row|
        @data.first.size.times do |col|
          new_mat.data[col][row] = @data[row][col]
        end
      end
      @data = new_mat.data
    end

    def t
      transpose
    end

    def shape
      return [@data.size, @data.first.size]
    end

    def reshape_new(width : Int32, height : Int32)
      _temp = 0.0
      temp = pointerof(_temp)
      @data = Slice.new(height) { |row| row = Slice.new(width) { temp } }
    end

    def show
      @data.each do |row|
        _r = [] of Float64
        row.each { |v| _r << v.value.round(5) }
        r = _r.to_s
        puts r.gsub(",") { "\t" }
      end
    end

    def show(logger : Logger, msg : String, round : Int32 = 4)
      @data.each do |row|
        _r = [] of Float64
        row.each { |v| _r << v.value.round(round) }
        r = _r.to_s.gsub(",") { " \t" }
        msg += "\n  " + r
      end
      logger.info(msg)
    end

    # def clone
    #   shape = self.shape
    #   new_mat = PtrMatrix.new(shape[0], shape[1])

    #   @data.first.size.times do |col|
    #     @data.size.times do |row|
    #       new_mat.data[row][col] = @data[row][col]
    #     end
    #   end

    #   return new_mat
    # end
  end

  # # Linear algebra math for Pointer matrices# #

  # # vector elment-by-element multiplication (For pointers)
  # def self.vector_mult(array1 : Slice(Pointer),
  #                      array2 : Slice(Pointer))
  #   raise MathError.new("Vectors must be the same size to multiply!") if array1.size != array2.size

  #   new_vector = [] of Float64
  #   (0..array1.size - 1).each do |i|
  #     result = array1[i].value.to_f64 * array2[i].value.to_f64
  #     new_vector << result
  #   end
  #   new_vector
  # end

  # # vector elment-by-element addition (For pointers)
  # def self.vector_sum(array1 : Slice(Pointer),
  #                     array2 : Slice(Pointer))
  #   raise MathError.new("Vectors must be the same size to sum!") if array1.size != array2.size

  #   new_vector = [] of Float64
  #   (0..array1.size - 1).each do |i|
  #     result = array1[i].value.to_f64 + array2[i].value.to_f64
  #     new_vector << result
  #   end
  #   new_vector
  # end

  # # Matrix dot product (for pointer matrices)
  # # Updates the result in the given pointer matrix
  # def self.dot_product(mat1 : Slice(Slice(Pointer)),
  #                      mat2 : Slice(Slice(Pointer)),
  #                      mat_out : Slice(Slice(Pointer)))
  #   mat1.each_with_index do |r, i|
  #     raise MathError.new("mat1 rows mismatch (row_0_size: #{mat1.first.size}, row_#{i}_size: #{r.size})") if mat1.first.size != r.size
  #   end
  #   mat2.each_with_index do |r, i|
  #     raise MathError.new("mat2 rows mismatch (row_0_size: #{mat2.first.size}, row_#{i}_size: #{r.size})") if mat2.first.size != r.size
  #   end
  #   raise MathError.new("Matrix dimention error, mat1 rows must equal mat2 columns") if mat1.size != mat2.first.size

  #   mat2 = transpose(mat2)
  #   mat1.size.times do |v1|
  #     mat2.size.times do |v2|
  #       new_vector = vector_mult(array1: mat1[v1], array2: mat2[v2]) # , array_out: mat_out[v1])
  #       new_value = new_vector.reduce { |acc, i| acc + i }
  #       mat_out[v1][v2].value = new_value
  #     end
  #   end
  # end

  # def self.transpose(mat : Slice(Slice(Pointer)))
  #   temp = 0.0
  #   new_mat = Slice.new(mat.first.size) { |new_r| Slice.new(mat.size) { pointerof(temp) } }

  #   mat.first.size.times do |col|
  #     mat.size.times do |row|
  #       new_mat[col][row] = mat[row][col]
  #     end
  #   end

  #   return new_mat
  # end

  # ##################################################################

  # def self.show_matrix(mat : Slice(Slice(Pointer)))
  #   mat.each do |r|
  #     row = [] of Float64
  #     r.each do |v|
  #       row << v.value
  #     end
  #     p row
  #   end
  # end
  # end
end
