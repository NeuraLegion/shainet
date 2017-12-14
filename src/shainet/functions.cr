module SHAInet
  # Activation functions
  def sigmoid(value : Int32 | Float32 | Float64)
    result = (1.0/(1.0 + Math.exp(-value))).to_f64
    return result
  end

  def tanh(value : Int32 | Float32 | Float64)
    result = ((1.0 - Math.exp(-2*value))/(1.0 + Math.exp(-2*value))).to_f64
    return result
  end

  def relu(value : Int32 | Float32 | Float64)
    if value <= 0
      return (0).to_f64
    else
      return value.to_f64
    end
  end

  def l_relu(value : Int32 | Float32 | Float64, slope : Float64)
    if value <= 0
      return slope.to_f64*value
    else
      return value.to_f64
    end
  end

  # vector multiplication
  def vector_mult(array1 : Array(Float64), array2 : Array(Float64))
    if array1.size != array2.size
      puts "Vectors must be the same size to multiply!"
    else
      new_vector = [] of Float64
      (0..array1.size - 1).each do |x|
        result = array1[x]*array2[x]
        new_vector << result
      end
      return new_vector
    end
  end

  def vector_sum(array1 : Array(Float64), array2 : Array(Float64))
    if array1.size != array2.size
      puts "Vectors must be the same size to sum!"
    else
      new_vector = [] of Float64
      (0..array1.size - 1).each do |x|
        result = array1[x] + array2[x]
        new_vector << result
      end
      return new_vector
    end
  end

  # translate an array of strings to one-hot vector matrix and hash dictionary
  def normalize_stcv(payloads : Array(String))
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
