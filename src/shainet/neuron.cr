module SHAInet
  # Each type of neuron uses and propogates data differently
  NEURON_TYPES     = [:memory, :eraser, :amplifier, :fader, :sensor]
  ACTIVATION_TYPES = [:tanh, :sigmoid, :bp_sigmoid, :log_sigmoid, :relu, :l_relu]

  class Neuron
    property :n_type, :synapses_in, :synapses_out
    property :activation, :input_sum, :bias, :error, :sigma_prime

    def initialize(@n_type : Symbol)
      raise NeuralNetInitalizationError.new("Must choose currect neuron types, if you're not sure choose :memory as a standard neuron") if NEURON_TYPES.any? { |x| x == @n_type } == false
      @synapses_in = [] of Synapse
      @synapses_out = [] of Synapse
      @activation = Float64.new(0)   # Activation of neuron after squashing function (a)
      @input_sum = Float64.new(0)    # Sum of activations*weights from input neurons (z)
      @bias = rand(-1.0..1.0).to_f64 # Activation threshhold (b)
      @error = Float64.new(0)        # Error of the neuron
      @sigma_prime = Float64.new(0)  # derivative of input_sum based on activation function used (s')
    end

    # This is the forward propogation
    # Allows the neuron to absorb the activation from its' own input neurons through the synapses
    # Then, it sums the information and an activation function is applied to normalize the data
    def activate(activation_function : Symbol = :sigmoid) : Float64
      raise NeuralNetRunError.new("Propogation requires a valid activation function.") unless ACTIVATION_TYPES.includes?(activation_function)

      new_memory = Array(Float64).new
      @synapses_in.each do |n| # Claclulate activation from each incoming neuron with applied weights, returns Array(Float64)
        new_memory << n.propagate_forward
      end
      @input_sum = new_memory.reduce { |acc, i| acc + i } # Sum all the information from input neurons, returns Float64
      @input_sum += @bias                                 # Add neuron bias (activation threshold)
      case activation_function                            # Apply squashing function
      when :tanh
        @activation = SHAInet.tanh(@input_sum)
      when :sigmoid
        @activation = SHAInet.sigmoid(@input_sum)
      when :bp_sigmoid
        @activation = SHAInet.bp_sigmoid(@input_sum)
      when :log_sigmoid
        @activation = SHAInet.log_sigmoid(@input_sum)
      when :relu
        @activation = SHAInet.relu(@input_sum)
      when :l_relu
        @activation = SHAInet.l_relu(@input_sum, 0.2) # value of 0.2 is the slope for x<0
      else
        raise NeuralNetRunError.new("Propogation requires a valid activation function.")
      end
    end

    # This is the backward propogation
    # Allows the neuron to absorb the error from its' own target neurons through the synapses
    # Then, it sums the information and a derivative of the activation function is applied to normalize the data
    def error_prop(activation_function : Symbol = :tanh) : Float64
      new_errors = Array(Float64).new
      @synapses_out.each do |n| # Claculate error from each target neuron with applied weights, returns Array(Float64)
        new_errors << n.propagate_backwards
      end
      z = new_errors.reduce { |acc, i| acc + i } # Sum all the information from target neurons, returns Float64
      case activation_function                   # Apply squashing function
      when :tanh
        @sigma_prime = SHAInet.tanh_prime(z)
        @error = SHAInet.tanh(z)*@sigma_prime
      when :sigmoid
        @sigma_prime = SHAInet.sigmoid_prime(z)
        @error = SHAInet.sigmoid(z)*@sigma_prime
        # when :bp_sigmoid
        #   @error = SHAInet.bp_sigmoid(z)
        # when :log_sigmoid
        #   @error = SHAInet.log_sigmoid(z)
        # when :relu
        #   @error = SHAInet.relu(z)
        # when :l_relu
        #   @error = SHAInet.l_relu(z, 0.2) # value of 0.2 is the slope for x<0
      else
        raise NeuralNetRunError.new("Propogation requires a valid activation function.")
      end
    end

    def inspect
      pp @n_type
      pp @memory
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
