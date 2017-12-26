module SHAInet
  # Each type of neuron uses and propogates data differently
  NEURON_TYPES     = [:memory, :eraser, :amplifier, :fader, :sensor]
  ACTIVATION_TYPES = [:tanh, :sigmoid, :bp_sigmoid, :log_sigmoid, :relu, :l_relu]

  class Neuron
    property :n_type, :synapses_in, :synapses_out, activation : Float64, gradient : Float64, bias : Float64, prev_bias : Float64
    getter :input_sum, :sigma_prime
    property prev_gradient : Float64, prev_delta : Float64, prev_delta_b : Float64
    property m_current : Float64, v_current : Float64, m_prev : Float64, v_prev : Float64

    def initialize(@n_type : Symbol)
      raise NeuralNetInitalizationError.new("Must choose currect neuron types, if you're not sure choose :memory as a standard neuron") if NEURON_TYPES.any? { |x| x == @n_type } == false
      @synapses_in = [] of Synapse
      @synapses_out = [] of Synapse
      @activation = Float64.new(0)    # Activation of neuron after squashing function (a)
      @gradient = Float64.new(0)      # Error of the neuron, sometimes refered to as delta
      @bias = rand(-1..1).to_f64      # Activation threshhold (b)
      @prev_bias = rand(-1..1).to_f64 # Needed for delta rule improvement using momentum

      @input_sum = Float64.new(0)   # Sum of activations*weights from input neurons (z)
      @sigma_prime = Float64.new(1) # derivative of input_sum based on activation function used (s')

      # Parameters needed for Rprop
      @prev_gradient = rand(-0.1..0.1).to_f64
      @prev_delta = rand(0.0..0.1).to_f64
      @prev_delta_b = rand(-0.1..0.1).to_f64

      # Parameters needed for Adam
      @m_current = Float64.new(0) # Current moment value
      @v_current = Float64.new(0) # Current moment**2 value
      @m_prev = Float64.new(0)    # Previous moment value
      @v_prev = Float64.new(0)    # Previous moment**2 value
    end

    # This is the forward propogation
    # Allows the neuron to absorb the activation from its' own input neurons through the synapses
    # Then, it sums the information and an activation function is applied to normalize the data
    def activate(activation_function : Symbol = :sigmoid) : Float64
      raise NeuralNetRunError.new("Propogation requires a valid activation function.") unless ACTIVATION_TYPES.includes?(activation_function)

      new_memory = Array(Float64).new
      @synapses_in.each do |synapse| # Claclulate activation from each incoming neuron with applied weights, returns Array(Float64)
        new_memory << synapse.propagate_forward
      end
      @input_sum = new_memory.reduce { |acc, i| acc + i } # Sum all the information from input neurons, returns Float64
      @input_sum += @bias                                 # Add neuron bias (activation threshold)
      case activation_function                            # Apply squashing function
      when :tanh
        @activation = SHAInet.tanh(@input_sum)
        @sigma_prime = SHAInet.tanh_prime(@input_sum) # Activation function derivative
      when :sigmoid
        @activation = SHAInet.sigmoid(@input_sum)
        @sigma_prime = SHAInet.sigmoid_prime(@input_sum)
      when :bp_sigmoid
        @activation = SHAInet.bp_sigmoid(@input_sum)
        @sigma_prime = SHAInet.bp_sigmoid_prime(@input_sum)
      when :log_sigmoid
        @activation = SHAInet.log_sigmoid(@input_sum)
        @sigma_prime = SHAInet.log_sigmoid_prime(@input_sum)
      when :relu
        @activation = SHAInet.relu(@input_sum)
        @sigma_prime = SHAInet.relu_prime(@input_sum)
      when :l_relu
        @activation = SHAInet.l_relu(@input_sum, 0.01) # value of 0.01 is the slope for x<0
        @sigma_prime = SHAInet.l_relu_prime(@input_sum)
      else
        raise NeuralNetRunError.new("Propogation requires a valid activation function.")
      end
    end

    # This is the backward propogation of the hidden layers
    # Allows the neuron to absorb the error from its' own target neurons through the synapses
    # Then, it sums the information and a derivative of the activation function is applied to normalize the data
    def hidden_error_prop : Float64
      new_errors = [] of Float64
      @synapses_out.each do |synapse| # Calculate weighted error from each target neuron, returns Array(Float64)
        new_errors << synapse.propagate_backward
      end
      weighted_error_sum = new_errors.reduce { |acc, i| acc + i } # Sum weighted error from target neurons (instead of using w_matrix*delta), returns Float64
      @gradient = weighted_error_sum*@sigma_prime                 # New error of the neuron
    end

    def inspect
      pp @n_type
      pp @activation
      pp @gradient
      pp @sigma_prime
      pp @synapses_in
      pp @synapses_out
    end

    def randomize_bias
      @bias = rand(-1.0..1.0).to_f64
    end

    def update_bias(value : Float64)
      @bias = value
    end
  end
end
