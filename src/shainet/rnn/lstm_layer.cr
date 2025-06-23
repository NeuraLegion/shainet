module SHAInet
  class LSTMLayer < Layer
    property hidden_state : Array(Float64)
    property cell_state : Array(Float64)
    property recurrent_synapses : Array(Array(Synapse))
    property input_bias : Array(Float64)
    property forget_bias : Array(Float64)
    property output_bias : Array(Float64)

    def initialize(n_type : String, l_size : Int32, activation_function : ActivationFunction = SHAInet.tanh)
      super(n_type, l_size, activation_function)
      @hidden_state = Array(Float64).new(l_size, 0.0)
      @cell_state = Array(Float64).new(l_size, 0.0)
      @recurrent_synapses = Array(Array(Synapse)).new(l_size) { Array(Synapse).new }
      @input_bias = Array(Float64).new(l_size) { rand(-0.1_f64..0.1_f64) }
      @forget_bias = Array(Float64).new(l_size) { rand(-0.1_f64..0.1_f64) }
      @output_bias = Array(Float64).new(l_size) { rand(-0.1_f64..0.1_f64) }

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
      @cell_state.map! { 0.0 }
      @neurons.each { |n| n.activation = 0.0 }
    end

    def activate_step
      new_hidden = Array(Float64).new(@l_size, 0.0)
      new_cell = Array(Float64).new(@l_size, 0.0)

      @neurons.each_with_index do |neuron, i|
        sum_in = neuron.bias
        sum_gate = 0.0
        sum_forget = 0.0
        sum_out = 0.0

        neuron.synapses_in.each do |syn|
          val = if @recurrent_synapses[i].includes?(syn)
                   j = @neurons.index(syn.source_neuron).not_nil!
                   @hidden_state[j]
                 else
                   syn.source_neuron.activation
                 end
          sum_in += val * syn.weight
          sum_gate += val * syn.weight
          sum_forget += val * syn.weight
          sum_out += val * syn.weight
        end

        gate_i, _ = SHAInet.sigmoid.call(sum_gate + @input_bias[i])
        gate_f, _ = SHAInet.sigmoid.call(sum_forget + @forget_bias[i])
        gate_o, _ = SHAInet.sigmoid.call(sum_out + @output_bias[i])
        cell_in, _ = @activation_function.call(sum_in)

        c = gate_f * @cell_state[i] + gate_i * cell_in
        h = gate_o * Math.tanh(c)

        neuron.input_sum = h
        neuron.sigma_prime = 1.0
        new_cell[i] = c
        new_hidden[i] = h
      end

      @neurons.each_with_index do |neuron, i|
        neuron.activation = new_hidden[i]
      end
      @hidden_state = new_hidden
      @cell_state = new_cell
      new_hidden
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
