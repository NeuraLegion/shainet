module SHAInet
  class LSTMLayer < Layer
    # Percentage of units to drop (0-100)
    property drop_percent : Int32
    property hidden_state : Array(Float64)
    property cell_state : Array(Float64)
    property recurrent_synapses : Array(Array(Synapse))
    property input_bias : Array(Float64)
    property forget_bias : Array(Float64)
    property output_bias : Array(Float64)
    property input_weights : Array(Array(Float64))
    property forget_weights : Array(Array(Float64))
    property output_weights : Array(Array(Float64))
    property input_w_grad : Array(Array(Float64))
    property forget_w_grad : Array(Array(Float64))
    property output_w_grad : Array(Array(Float64))
    property input_b_grad : Array(Float64)
    property forget_b_grad : Array(Float64)
    property output_b_grad : Array(Float64)
    @gate_setup : Bool = false

    def initialize(n_type : String, l_size : Int32,
                   activation_function : ActivationFunction = SHAInet.tanh,
                   drop_percent : Int32 = 0)
      super(n_type, l_size, activation_function)
      @drop_percent = drop_percent
      @hidden_state = Array(Float64).new(l_size, 0.0)
      @cell_state = Array(Float64).new(l_size, 0.0)
      @recurrent_synapses = Array(Array(Synapse)).new(l_size) { Array(Synapse).new }
      @input_bias = Array(Float64).new(l_size) { rand(-0.1_f64..0.1_f64) }
      @forget_bias = Array(Float64).new(l_size) { rand(-0.1_f64..0.1_f64) }
      @output_bias = Array(Float64).new(l_size) { rand(-0.1_f64..0.1_f64) }
      @input_weights = Array(Array(Float64)).new(l_size) { Array(Float64).new }
      @forget_weights = Array(Array(Float64)).new(l_size) { Array(Float64).new }
      @output_weights = Array(Array(Float64)).new(l_size) { Array(Float64).new }
      @input_w_grad = Array(Array(Float64)).new(l_size) { Array(Float64).new }
      @forget_w_grad = Array(Array(Float64)).new(l_size) { Array(Float64).new }
      @output_w_grad = Array(Array(Float64)).new(l_size) { Array(Float64).new }
      @input_b_grad = Array(Float64).new(l_size, 0.0)
      @forget_b_grad = Array(Float64).new(l_size, 0.0)
      @output_b_grad = Array(Float64).new(l_size, 0.0)

      @neurons.each_with_index do |dest_neuron, i|
        @neurons.each_with_index do |src_neuron, j|
          syn = Synapse.new(src_neuron, dest_neuron)
          src_neuron.synapses_out << syn
          dest_neuron.synapses_in << syn
          @recurrent_synapses[i] << syn
        end
      end
    end

    def setup_gate_params
      return if @gate_setup
      @neurons.each_with_index do |neuron, i|
        syn_count = neuron.synapses_in.size
        @input_weights[i] = Array(Float64).new(syn_count) { rand(-0.1_f64..0.1_f64) }
        @forget_weights[i] = Array(Float64).new(syn_count) { rand(-0.1_f64..0.1_f64) }
        @output_weights[i] = Array(Float64).new(syn_count) { rand(-0.1_f64..0.1_f64) }
        @input_w_grad[i] = Array(Float64).new(syn_count, 0.0)
        @forget_w_grad[i] = Array(Float64).new(syn_count, 0.0)
        @output_w_grad[i] = Array(Float64).new(syn_count, 0.0)
      end
      @gate_setup = true
    end

    def zero_gate_gradients
      @input_w_grad.each &.map! { 0.0 }
      @forget_w_grad.each &.map! { 0.0 }
      @output_w_grad.each &.map! { 0.0 }
      @input_b_grad.map! { 0.0 }
      @forget_b_grad.map! { 0.0 }
      @output_b_grad.map! { 0.0 }
    end

    def accumulate_gate_gradients
      @neurons.each_with_index do |neuron, i|
        neuron.synapses_in.each_with_index do |syn, j|
          grad = syn.source_neuron.activation * neuron.gradient
          @input_w_grad[i][j] += grad
          @forget_w_grad[i][j] += grad
          @output_w_grad[i][j] += grad
        end
        @input_b_grad[i] += neuron.gradient
        @forget_b_grad[i] += neuron.gradient
        @output_b_grad[i] += neuron.gradient
      end
    end

    def update_gate_params(lr : Float64)
      @input_weights.each_with_index do |row, i|
        row.each_index do |j|
          row[j] -= lr * @input_w_grad[i][j]
          @forget_weights[i][j] -= lr * @forget_w_grad[i][j]
          @output_weights[i][j] -= lr * @output_w_grad[i][j]
        end
      end
      @input_bias.each_index do |i|
        @input_bias[i] -= lr * @input_b_grad[i]
        @forget_bias[i] -= lr * @forget_b_grad[i]
        @output_bias[i] -= lr * @output_b_grad[i]
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

        neuron.synapses_in.each_with_index do |syn, j|
          val = if @recurrent_synapses[i].includes?(syn)
                  j2 = @neurons.index(syn.source_neuron).not_nil!
                  @hidden_state[j2]
                else
                  syn.source_neuron.activation
                end
          sum_in += val * syn.weight
          sum_gate += val * @input_weights[i][j]
          sum_forget += val * @forget_weights[i][j]
          sum_out += val * @output_weights[i][j]
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

      new_hidden = RNNDropout.apply(new_hidden, @drop_percent) if @drop_percent > 0

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
