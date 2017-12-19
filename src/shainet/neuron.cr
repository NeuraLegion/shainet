module SHAInet
  # Each type of neuron uses and propogates data differently
  NEURON_TYPES     = [:memory, :eraser, :amplifier, :fader, :sensor]
  ACTIVATION_TYPES = [:tanh, :sigmoid, :bp_sigmoid, :log_sigmoid, :relu, :l_relu]

  class Neuron
    property :n_type, :synapses_in, :synapses_out
    property memory : Float64, error : Float64 ,bias : Float64, 

    def initialize(@n_type : Symbol)
      raise NeuralNetInitalizationError.new("Must choose currect neuron types, if you're not sure choose :memory as a standard neuron") if NEURON_TYPES.any? { |x| x == @n_type } == false
      @synapses_in = [] of Synapse
      @synapses_out = [] of Synapse
      @memory = Float64.new(0)
      @error = Float64.new(0)
      @bias = rand(-1.0..1.0)
    end

    # This is the forward propogation
    # Allows the neuron to absorb the memory from its' own input neurons through the synapses
    # Then, it sums the information and an activation function is applied to normalize the data
    def activate(activation_function : Symbol = :sigmoid) : Float64
      raise NeuralNetRunError.new("Propogation requires a valid activation function.") unless ACTIVATION_TYPES.includes?(activation_function)

      new_memory = Array(Float64).new
      @synapses_in.each do |n|            # Claclulate memory from each incoming neuron with applied weights, returns Array(Float64)
        new_memory << n.propagate_forward 
      end
      output = new_memory.reduce { |acc, i| acc + i } # Sum all the information from input neurons, returns Float64
      output += @bias                                 # Add neuron bias (activation threshold)
      case activation_function                        # Apply squashing function
      when :tanh
        @memory = SHAInet.tanh(output)
      when :sigmoid
        @memory = SHAInet.sigmoid(output)
      when :bp_sigmoid
        @memory = SHAInet.bp_sigmoid(output)
      when :log_sigmoid
        @memory = SHAInet.log_sigmoid(output)
      when :relu
        @memory = SHAInet.relu(output)
      when :l_relu
        @memory = SHAInet.l_relu(output, 0.2) # value of 0.2 is the slope for x<0
      else
        raise NeuralNetRunError.new("Propogation requires a valid activation function.")
      end
    end

    # This is the backward propogation
    # Allows the neuron to absorb the error from its' own target neurons through the synapses
    # Then, it sums the information and an activation function is applied to normalize the data
    def error_prop(activation_function : Symbol = :tanh) : Float64
      new_error = Array(Float64).new
      @synapses_out.each do |n|            # Claculate error from each target neuron with applied weights, returns Array(Float64)
        new_error << n.propagate_backwards 
      end
      output = new_error.reduce { |acc, i| acc + i }  # Sum all the information from target neurons, returns Float64
      case activation_function                        # Apply squashing function
      when :tanh
        @error = SHAInet.tanh(output)
      when :sigmoid
        @error = SHAInet.sigmoid(output)
      when :bp_sigmoid
        @error = SHAInet.bp_sigmoid(output)
      when :log_sigmoid
        @error = SHAInet.log_sigmoid(output)
      when :relu
        @error = SHAInet.relu(output)
      when :l_relu
        @error = SHAInet.l_relu(output, 0.2) # value of 0.2 is the slope for x<0
      else
        raise NeuralNetRunError.new("Propogation requires a valid activation function.")
      end
    end
    end

    def inspect
      pp @n_type
      pp @memory
      pp @synapses_in
      pp @synapses_out
      pp @output
    end

    def randomize_bias
      @bias = rand(-1.0..1.0).to_f64
    end

    def update_bias(value : Float64)
      @bias = value
    end


    # def derivative
    #   output * (1 - output)
    # end

    # def output_train(rate, target)
    #   @error = (target - output) * derivative
    #   update_weights(rate)
    # end

    # def hidden_train(rate)
    #   @error = synapses_out.reduce(0.0) do |sum, synapse|
    #     sum + synapse.prev_weight * synapse.dest_neuron.error
    #   end * derivative
    #   update_weights(rate)
    # end

    # def update_weights(rate)
    #   synapses_in.each do |synapse|
    #     temp_weight = synapse.weight
    #     synapse.weight += (rate * LEARNING_RATE * error * synapse.source_neuron.output) + (MOMENTUM * (synapse.weight - synapse.prev_weight))
    #     synapse.prev_weight = temp_weight
    #   end
    #   temp_threshold = threshold
    #   @threshold += (rate * LEARNING_RATE * error * -1) + (MOMENTUM * (threshold - prev_threshold))
    #   @prev_threshold = temp_threshold
    # end
  end
end
