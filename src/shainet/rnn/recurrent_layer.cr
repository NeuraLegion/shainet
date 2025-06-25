module SHAInet
  class RecurrentLayer < Layer
    property hidden_state
    property recurrent_synapses : Array(Array(Synapse))

    def initialize(n_type : String, l_size : Int32, activation_function : ActivationFunction = SHAInet.sigmoid)
      super(n_type, l_size, activation_function)
      @hidden_state = Array(Float64).new(l_size, 0.0)
      @recurrent_synapses = Array(Array(Synapse)).new(l_size) { Array(Synapse).new }

      @neurons.each_with_index do |dest_neuron, i|
        @neurons.each_with_index do |src_neuron, j|
          syn = Synapse.new(src_neuron, dest_neuron)
          src_neuron.synapses_out << syn
          dest_neuron.synapses_in << syn
          @recurrent_synapses[i] << syn
        end
      end
    end

    def reset_state
      @hidden_state.map! { 0.0 }
      @neurons.each { |n| n.activation = 0.0 }
    end

    def activate_step
      new_state = Array(Float64).new(@l_size, 0.0)
      @neurons.each_with_index do |neuron, i|
        sum = neuron.bias
        neuron.synapses_in.each do |syn|
          if @recurrent_synapses[i].includes?(syn)
            j = @neurons.index(syn.source_neuron).not_nil!
            sum += @hidden_state[j] * syn.weight
          else
            sum += syn.source_neuron.activation * syn.weight
          end
        end
        neuron.input_sum = sum
        act, sp = @activation_function.call(sum)
        neuron.sigma_prime = sp
        new_state[i] = act
      end
      @neurons.each_with_index do |neuron, i|
        neuron.activation = new_state[i]
      end
      @hidden_state = new_state.clone
      new_state
    end

    def activate_sequence(sequence : Array(Array(GenNum)))
      outputs = [] of Array(Float64)
      sequence.each do |_|
        outputs << activate_step
      end
      outputs
    end

    def backprop_sequence
      @neurons.each { |neuron| neuron.hidden_error_prop }
    end
  end
end
