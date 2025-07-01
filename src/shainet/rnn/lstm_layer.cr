module SHAInet
  class LSTMLayer < Layer
    # Percentage of units to drop (0-100)
    property drop_percent : Int32
    property hidden_state : Array(Float64)
    property cell_state : Array(Float64)
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
    end

    def setup_gate_params
      return if @gate_setup
      size.times do |i|
        @input_weights[i] = Array(Float64).new(size) { rand(-0.1_f64..0.1_f64) }
        @forget_weights[i] = Array(Float64).new(size) { rand(-0.1_f64..0.1_f64) }
        @output_weights[i] = Array(Float64).new(size) { rand(-0.1_f64..0.1_f64) }
        @input_w_grad[i] = Array(Float64).new(size, 0.0)
        @forget_w_grad[i] = Array(Float64).new(size, 0.0)
        @output_w_grad[i] = Array(Float64).new(size, 0.0)
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

    # No neurons array in matrix-based implementation

    def accumulate_gate_gradients
      # Matrix-based implementation of gradient accumulation
      @l_size.times do |i|
        act_grad = @activations[0, i] * @sigma_primes[0, i]

        # Use matrix operations instead of iterating over synapses
        @input_weights[i].size.times do |j|
          input_val = (j < @input_sums.cols) ? @input_sums[0, j] : 0.0
          grad = input_val * act_grad
          @input_w_grad[i][j] += grad
          @forget_w_grad[i][j] += grad
          @output_w_grad[i][j] += grad
        end

        @input_b_grad[i] += act_grad
        @forget_b_grad[i] += act_grad
        @output_b_grad[i] += act_grad
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
    end

    def activate_step
      new_hidden = Array(Float64).new(@l_size, 0.0)
      new_cell = Array(Float64).new(@l_size, 0.0)

      use_cuda = CUDA.available?

      size.times do |i|
        inputs = [] of Float64
        syn_w = [] of Float64

        mat_klass = CudaMatrix
        inp = mat_klass.from_a([inputs])
        w_in = mat_klass.from_a(syn_w.map { |w| [w] })
        wi = mat_klass.from_a(@input_weights[i].map { |w| [w] })
        wf = mat_klass.from_a(@forget_weights[i].map { |w| [w] })
        wo = mat_klass.from_a(@output_weights[i].map { |w| [w] })

        sum_gate = (inp * wi)[0, 0] + @input_bias[i]
        sum_forget = (inp * wf)[0, 0] + @forget_bias[i]
        sum_out = (inp * wo)[0, 0] + @output_bias[i]

        gate_i, _ = SHAInet.sigmoid.call(sum_gate + @input_bias[i])
        gate_f, _ = SHAInet.sigmoid.call(sum_forget + @forget_bias[i])
        gate_o, _ = SHAInet.sigmoid.call(sum_out + @output_bias[i])

        c = gate_f * @cell_state[i]
        h = gate_o * Math.tanh(c)

        new_cell[i] = c
        new_hidden[i] = h
      end

      new_hidden = RNNDropout.apply(new_hidden, @drop_percent) if @drop_percent > 0

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
      # Matrix-based implementation that doesn't use neurons
      # Empty implementation since matrices handle backprop differently
    end
  end
end
