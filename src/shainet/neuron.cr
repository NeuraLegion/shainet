module SHAInet
  # Each type of neuron uses and propogates data differently
  NEURON_TYPES     = [:memory, :eraser, :amplifier, :fader, :sensor]
  ACTIVATION_TYPES = [:tanh, :sigmoid, :relu, :l_relu]

  class Neuron
    property :n_type, :memory_size, :memory, :synapses_in, :synapses_out, :output

    def initialize(@n_type : Symbol, @memory_size : Int32)
      raise NeuralNetInitalizationError.new("Must choose currect neuron types, if you're not sure choose :memory as a standard neuron") if NEURON_TYPES.any? { |x| x == @n_type } == false
      @synapses_in = [] of Synapse
      @synapses_out = [] of Synapse

      # Memory size is determined by total of possible choices per neuron
      @memory = Array(Float64).new(@memory_size) { |i| 0.0 }
      @output = Array(Float64).new
    end

    # Allows the neuron to absorbs information from its' own input neurons through the synapses
    # Then, it sums the information and an activation function is applied to normalize the data
    def learn(activation_function : Symbol = :sigmoid) : Array(Float64)
      raise NeuralNetRunError.new("Propogation requires a valid activation function.") if ACTIVATION_TYPES.any? { |x| x == activation_function } == false

      new_memory = Array(Array(Float64)).new
      @synapses_in.each do |x|
        new_memory << x.propagate # Array(Float64)
      end
      puts "New mem: #{new_memory}"
      output = new_memory.transpose.map(&.sum)
      puts "Output: #{output}"
      case activation_function
      when :tanh
        @memory = output.map { |f| SHAInet.tanh(f) }
      when :sigmoid
        @memory = output.map { |f| SHAInet.sigmoid(f) }
        puts @memory
        @memory
      when :relu
        @memory = output.map { |f| SHAInet.relu(f) }
      when :l_relu
        @memory = output.map { |f| SHAInet.l_relu(f, 0.2) }
      else
        raise NeuralNetRunError.new("Propogation requires a valid activation function.")
      end
    end

    def inspect
      pp @n_type
      pp @memory
      pp @synapses_in
      pp @synapses_out
      pp @output
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
