module SHAInet
  alias ActivationFunction = Proc(GenNum, Tuple(Float64, Float64))
  alias CostFunction = Proc(GenNum, GenNum, NamedTuple(value: Float64, derivative: Float64))

  # As Procs

  def self.none : ActivationFunction # Output range -inf..inf)
    ->(value : GenNum) { {value.to_f64, 1.0_f64} }
  end

  def self.sigmoid : ActivationFunction # Output range (0..1)
    ->(value : GenNum) { {_sigmoid(value), _sigmoid_prime(value)} }
  end

  def self.bp_sigmoid : ActivationFunction # Output range (-1..1)
    ->(value : GenNum) { {_bp_sigmoid(value), _bp_sigmoid_prime(value)} }
  end

  def self.log_sigmoid : ActivationFunction # Output range (0..1)
    ->(value : GenNum) { {_log_sigmoid(value), _log_sigmoid_prime(value)} }
  end

  def self.tanh : ActivationFunction # Output range (-1..1)
    ->(value : GenNum) { {_tanh(value), _tanh_prime(value)} }
  end

  def self.relu : ActivationFunction # Output range (0..inf)
    ->(value : GenNum) { {_relu(value), _relu_prime(value)} }
  end

  def self.l_relu : ActivationFunction # Output range (-inf..inf)
    # (value : GenNum, slope : Float64 = 0.01) : Float64
    ->(value : GenNum) { {_l_relu(value), _l_relu_prime(value)} }
  end

  # # Activation functions # #

  def self._sigmoid(value : GenNum) : Float64 # Output range (0..1)
    (1.0/(1.0 + Math::E**(-value))).to_f64
  end

  def self._bp_sigmoid(value : GenNum) : Float64 # Output range (-1..1)
    ((1.0 - Math::E**(-value))/(1.0 + Math::E**(-value))).to_f64
  end

  def self._log_sigmoid(value : GenNum) : Float64 # Output range (0..1)
    ((Math::E**(value))/(1.0 + Math::E**(value))).to_f64
  end

  def self._tanh(value : GenNum) : Float64 # Output range (-1..1)
    ((Math::E**(value) - Math::E**(-value))/(Math::E**(value) + Math::E**(-value))).to_f64
  end

  def self._relu(value : GenNum) # Output range (0..inf)
    if value < 0
      (0).to_f64
    else
      value.to_f64
    end
  end

  def self._l_relu(value : GenNum, slope : Float64 = 0.01) : Float64 # Output range (-inf..inf)
    if value < 0
      slope.to_f64*value.to_f64
    else
      value.to_f64
    end
  end

  def self.softmax(array : Array(GenNum)) : Array(Float64)
    out_array = Array(Float64).new(array.size) { 0.0 }
    exp_sum = Float64.new(0.0)
    array.each { |value| exp_sum += Math::E**(value) }
    array.size.times { |i| out_array[i] += (Math::E**array[i])/exp_sum }
    out_array
  end

  # The input array in this case has to be the output array of the softmax function
  def self.softmax_prime(array : Array(GenNum)) : Array(Float64)
    out_array = Array(Float64).new(array.size) { 0.0 }
    array.each_with_index { |value, i| out_array[i] = array[i]*(1 - array[i]) }
    out_array
  end

  # Not working yet, do not use
  def self.log_softmax(array : Array(GenNum)) : Array(Float64)
    out_array = Array(Float64).new(array.size) { 0.0 }
    m = array.max # Max exponent from input array
    exp_sum = Float64.new(0.0)
    array.each { |value| exp_sum += Math::E**(value - m) }

    array.size.times { |i| out_array[i] = (Math::E**(array[i] - m - Math.log(exp_sum, 10))) }
    out_array
  end

  # # Derivatives of activation functions # #

  def self._sigmoid_prime(value : GenNum) : Float64
    _sigmoid(value)*(1.0 - _sigmoid(value)).to_f64
  end

  def self._bp_sigmoid_prime(value : GenNum) : Float64
    (2*Math::E**(value)/(Math::E**(value) + 1)**2).to_f64
  end

  def self._log_sigmoid_prime(value : GenNum) : Float64
    (Math::E**(value)/(Math::E**(value) + 1)**2).to_f64
  end

  def self._tanh_prime(value : GenNum) : Float64
    (1.0 - _tanh(value)**2).to_f64
  end

  def self._relu_prime(value : GenNum) : Float64
    if value < 0
      (0).to_f64
    else
      (1).to_f64
    end
  end

  def self._l_relu_prime(value : GenNum, slope : Float64 = 0.01) : Float64
    if value < 0
      slope
    else
      (1).to_f64
    end
  end

  ##################################################################
  # # Procs for cost functions

  def self.quadratic_cost : CostFunction
    ->(expected : GenNum, actual : GenNum) {
      {value:      _quadratic_cost(expected.to_f64, actual.to_f64),
       derivative: _quadratic_cost_derivative(expected.to_f64, actual.to_f64)}
    }
  end

  def self.cross_entropy_cost : CostFunction
    ->(expected : GenNum, actual : GenNum) {
      {value:      _cross_entropy_cost(expected.to_f64, actual.to_f64),
       derivative: _cross_entropy_cost_derivative(expected.to_f64, actual.to_f64)}
    }
  end

  # # Cost functions  # #

  def self._quadratic_cost(expected : Float64, actual : Float64) : Float64
    (0.5*(actual - expected)**2).to_f64
  end

  def self._cross_entropy_cost(expected : Float64, actual : Float64) : Float64
    # raise MathError.new("Cross entropy cost is not implemented fully yet, please use quadratic cost for now.")
    if expected == 1.0
      if actual <= 0.000001
        10.0
      elsif actual == 1.0
        0.0
      else
        (-1)*Math.log((actual), Math::E)
      end
    elsif expected == 0.0
      if actual >= 0.999999
        10.0
      elsif actual == 0.0
        0.0
      else
        (-1)*Math.log((1.0 - actual), Math::E)
      end
    else
      raise MathError.new("Expected value must be 0 or 1 for cross entropy cost.")
    end
  end

  # # Derivatives of cost functions # #
  def self._quadratic_cost_derivative(expected : Float64, actual : Float64) : Float64
    (actual - expected).to_f64
  end

  def self._cross_entropy_cost_derivative(expected : Float64, actual : Float64) : Float64
    (actual - expected).to_f64
  end

  ##################################################################

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

  ##################################################################

  # # Other # #

  # Used in Rprop
  def self.sign(input : GenNum)
    if input > 0
      +1
    elsif input < 0
      -1
    else
      0
    end
  end
end
