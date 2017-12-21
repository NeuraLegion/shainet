require "matrix_extend"

module SHAInet
  include MatrixExtend

  # # Activation functions # #

  def self.sigmoid(value : Int32 | Float32 | Float64) # Output range (0..1)
    (1.0/(1.0 + Math.exp(-value))).to_f64
  end

  def self.bp_sigmoid(value : Int32 | Float32 | Float64) # Output range (-1..1)
    ((1.0 - Math.exp(-value))/(1.0 + Math.exp(-value))).to_f64
  end

  def self.log_sigmoid(value : Int32 | Float32 | Float64) # Output range (0..1)
    ((Math.exp(value))/(1.0 + Math.exp(value))).to_f64
  end

  def self.tanh(value : Int32 | Float32 | Float64) # Output range (-1..1)
    (((Math.exp(value)) - Math.exp(-value))/(Math.exp(value)) + Math.exp(-value)).to_f64
  end

  def self.relu(value : Int32 | Float32 | Float64) # Output range (0..inf)
    if value <= 0
      (0).to_f64
    else
      value.to_f64
    end
  end

  def self.l_relu(value : Int32 | Float32 | Float64, slope : Float64) # Output range (-inf..inf)
    if value <= 0
      slope.to_f64*value.to_f64
    else
      value.to_f64
    end
  end

  # # Derivatives of activation functions # #

  def self.sigmoid_prime(value : Float64)
    sigmoid(value)*(1 - sigmoid(value))
  end

  def self.tanh_prime(value : Float64)
    (1 - tanh(value)**2)
  end

  # # Cost functions for a single point value (slope at that point based on the function) ##

  def self.quadratic_cost_derivative(expected : Float64, actual : Float64) : Float64
    # Cost function = 0.5*(actual - expected)**2
    return (actual - expected) # Slope at a single point
  end

  def self.cross_entropy_cost_derivative(expected : Float64, actual : Float64) : Float64
    # Cost function = (-1)*(expected*Math.log((actual), Math::E) + (1 - expected)*Math.log((1 - actual), Math::E))
    return ((actual - expected)/((1 - actual)*actual)) # Slope at a single point
  end

  # # Linear algebra math # #

  # Matrix multiplication
  def self.dot_product(matrix1 : Array(Array(Int32 | Float32 | Float64)), matrix2 : Array(Array(Int32 | Float32 | Float64)))
    if matrix1.all? { |row| row.size == matrix1.first.size } == false
      raise MathError.new("Matrix1 has rows of different size")
    elsif matrix2.all? { |row| row.size == matrix2.first.size } == false
      raise MathError.new("Matrix2 has rows of different size")
    elsif matrix1.first.size != matrix2.size
      raise MathError.new("Matrix1 row dimention must equal matrix2 col dimention")
    end
    new_matrix = Array(Array(Float64)).new
    matrix1.each do |row|
      new_row = [] of Float64 | Float32 | Int32
      matrix2.each do |col|
        sum = [] of Float64 | Float32 | Int32
        row.each { |rv| col.each { |cv| sum << rv*cv } }
        new_row << sum
      end
      new_matrix << new_row
    end
    return new_matrix
  end

  # Element-wise multiplications
  def h_product
    # TODO
  end

  # vector multiplication
  def self.vector_mult(array1 : Array(Float64), array2 : Array(Float64))
    raise MathError.new("Vectors must be the same size to multiply!") if array1.size != array2.size

    new_vector = [] of Float64
    (0..array1.size - 1).each do |x|
      result = array1[x]*array2[x]
      new_vector << result
    end
    new_vector
  end

  def self.vector_sum(array1 : Array(Float64), array2 : Array(Float64))
    raise MathError.new("Vectors must be the same size to sum!") if array1.size != array2.size

    new_vector = [] of Float64
    (0..array1.size - 1).each do |x|
      result = array1[x] + array2[x]
      new_vector << result
    end
    new_vector
  end

  # # Data manipulation # #

  # translate an array of strings to one-hot vector matrix and hash dictionary
  def self.normalize_stcv(payloads : Array(String))
    s = payloads.max_by &.size # Find biggest string, all strings will be padded to its' size
    input_size = s.size
    payloads_c = Array(Array(String)).new
    payloads_v = Array(Array(Array(Int32))).new
    vocabulary = [] of String
    vocabulary_v = Hash(String, Array(Int32)).new

    # Split payloads and update vocabulary
    payloads.each do |str|
      x = str.split("")

      # add new unique chars to vocabulary
      x.each { |char| vocabulary << char }

      # save strings as arrays of chars
      payloads_c << x
    end

    # create hash of char-to-vector (vector size = all possible chars)
    vocabulary.uniq!
    (0..vocabulary.size - 1).each do |x|
      char_v = Array(Int32).new
      (0..vocabulary.size - 1).each do |i|
        if i == x
          char_v << 1
        else
          char_v << 0
        end
      end
      vocabulary_v[vocabulary[x]] = char_v
    end
    zero_v = Array.new(vocabulary.size) { |i| 0 }

    # Translate the strings into arrays of char-vectors
    payloads_c.each do |str|
      str_v = Array(Array(Int32)).new
      str.each { |char| str_v << vocabulary_v[char] }
      payloads_v << str_v
    end

    # Pad all string vectors with 0 vectors for uniform input size
    payloads_v.each do |str|
      while str.size < input_size
        str << zero_v
      end
    end

    return input_size, vocabulary_v, payloads_v
  end
end
