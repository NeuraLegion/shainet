require "logger"

module SHAInet
  class ConvLayer
    getter prev_layer : CNNLayer | ConvLayer, filters : Array(Filter)
    getter window_size : Int32, stride : Int32, padding : Int32, activation_function : Proc(GenNum, Array(Float64))
    getter learning_rate : Float64, momentum : Float64

    #################################################
    # # This part is for dealing with conv layers # #

    def initialize(@prev_layer : ConvLayer | CNNLayer,
                   filters_num : Int32 = 1,
                   @window_size : Int32 = 1,
                   @stride : Int32 = 1,
                   @padding : Int32 = 0,
                   @activation_function : Proc(GenNum, Array(Float64)) = SHAInet.none,
                   @learning_rate : Float64 = 0.7,
                   @momentum : Float64 = 0.3,
                   @logger : Logger = Logger.new(STDOUT))
      #
      raise CNNInitializationError.new("ConvLayer must have at least one filter") if filters_num < 1
      raise CNNInitializationError.new("Padding value must be Int32 >= 0") if @padding < 0
      raise CNNInitializationError.new("Window size value must be Int32 >= 1") if @window_size < 1
      raise CNNInitializationError.new("Stride value must be Int32 >= 1") if @stride < 1

      filters = @prev_layer.filters.size             # In conv layers channels are replaced by the feature maps,stored in the Filter class
      width = @prev_layer.filters.first.neurons.size # Assumes row == height

      # This is a calculation to make sure the input volume matches a correct desired output volume
      output_width = ((width - @window_size + 2*@padding)/@stride + 1)
      unless output_width.to_i == output_width
        raise CNNInitializationError.new("Output volume must be a whole number, change: window size, stride and/or padding")
      end

      @filters = Array(Filter).new(filters_num) { Filter.new([output_width.to_i, output_width.to_i, filters], @padding, @window_size, @stride, @activation_function) }

      @w_gradient = Array(Float64).new # Needed for batch train
      @b_gradient = Array(Float64).new # Needed for batch train

    end

    # Use each filter to create feature maps from the input data of the previous layer
    def activate
      @filters.each { |filter| filter.propagate_forward(@prev_layer) }
    end

    def error_prop
      @filters.each { |filter| filter.propagate_backward(@prev_layer) }
    end

    def update_wb(learn_type : Symbol | String, batch : Bool = false)
      @filters.each do |filter|
        filter.synapses.size.times do |channel|
          filter.synapses[channel].size.times do |row|
            filter.synapses[channel][row].size.times do |col|
              synapse = filter.synapses[channel][row][col]
              # Get current gradient
              if batch == true
                raise CNNInitializationError.new("Batch is not implemented yet.")
                # synapse.gradient = @w_gradient.not_nil![i]
              end

              case learn_type.to_s
              # Update weights based on the gradients and delta rule (including momentum)
              when "sgdm"
                delta_weight = (-1)*@learning_rate*synapse.gradient_sum + @momentum*(synapse.weight - synapse.prev_weight)
                synapse.weight += delta_weight
                synapse.gradient_sum = Float64.new(0)
                synapse.prev_weight = synapse.weight

                # Update weights based on Resilient backpropogation (Rprop), using the improved varient iRprop+
              when "rprop"
                raise CNNInitializationError.new("rProp is not implemented yet.")
                # if synapse.prev_gradient*synapse.gradient > 0
                #   delta = [@etah_plus*synapse.prev_delta, @delta_max].min
                #   delta_weight = (-1)*SHAInet.sign(synapse.gradient)*delta

                #   synapse.weight += delta_weight
                #   synapse.prev_weight = synapse.weight
                #   synapse.prev_delta = delta
                #   synapse.prev_delta_w = delta_weight
                # elsif synapse.prev_gradient*synapse.gradient < 0.0
                #   delta = [@etah_minus*synapse.prev_delta, @delta_min].max

                #   synapse.weight -= synapse.prev_delta_w if @mean_error >= @prev_mean_error

                #   synapse.prev_gradient = 0.0
                #   synapse.prev_delta = delta
                # elsif synapse.prev_gradient*synapse.gradient == 0.0
                #   delta_weight = (-1)*SHAInet.sign(synapse.gradient)*synapse.prev_delta

                #   synapse.weight += delta_weight
                #   synapse.prev_delta = @delta_min
                #   synapse.prev_delta_w = delta_weight
                # end

                # Update weights based on Adaptive moment estimation (Adam)
              when "adam"
                raise CNNInitializationError.new("ADAM is not implemented yet.")
                # synapse.m_current = @beta1*synapse.m_prev + (1 - @beta1)*synapse.gradient
                # synapse.v_current = @beta2*synapse.v_prev + (1 - @beta2)*(synapse.gradient)**2

                # m_hat = synapse.m_current/(1 - (@beta1)**@time_step)
                # v_hat = synapse.v_current/(1 - (@beta2)**@time_step)
                # synapse.weight -= (@alpha*m_hat)/(v_hat**0.5 + @epsilon)

                # synapse.m_prev = synapse.m_current
                # synapse.v_prev = synapse.v_current
              end
            end
          end
        end

        # Update biases of the layer

        if batch == true
          raise CNNInitializationError.new("Batch is not implemented yet.")
          # neuron.gradient = @b_gradient.not_nil![i]
        else
          filter.bias = filter.bias_sum
          filter.bias_sum = Float64.new(0)
        end

        case learn_type.to_s
        # Update biases based on the gradients and delta rule (including momentum)
        when "sgdm"
          delta_bias = (-1)*@learning_rate*(filter.bias) + @momentum*(filter.bias - filter.prev_bias)
          filter.bias += delta_bias
          filter.prev_bias = filter.bias

          # Update weights based on Resilient backpropogation (Rprop), using the improved varient iRprop+
        when "rprop"
          raise CNNInitializationError.new("rProp is not implemented yet.")
          # if neuron.prev_gradient*neuron.gradient > 0
          #   delta = [@etah_plus*neuron.prev_delta, @delta_max].min
          #   delta_bias = (-1)*SHAInet.sign(neuron.gradient)*delta

          #   neuron.bias += delta_bias
          #   neuron.prev_bias = neuron.bias
          #   neuron.prev_delta = delta
          #   neuron.prev_delta_b = delta_bias
          # elsif neuron.prev_gradient*neuron.gradient < 0.0
          #   delta = [@etah_minus*neuron.prev_delta, @delta_min].max

          #   neuron.bias -= neuron.prev_delta_b if @mean_error >= @prev_mean_error

          #   neuron.prev_gradient = 0.0
          #   neuron.prev_delta = delta
          # elsif neuron.prev_gradient*neuron.gradient == 0.0
          #   delta_bias = (-1)*SHAInet.sign(neuron.gradient)*@delta_min*neuron.prev_delta

          #   neuron.bias += delta_bias
          #   neuron.prev_delta = @delta_min
          #   neuron.prev_delta_b = delta_bias
          # end

          # Update weights based on Adaptive moment estimation (Adam)
        when "adam"
          raise CNNInitializationError.new("ADAM is not implemented yet.")
          # neuron.m_current = @beta1*neuron.m_prev + (1 - @beta1)*neuron.gradient
          # neuron.v_current = @beta2*neuron.v_prev + (1 - @beta2)*(neuron.gradient)**2

          # m_hat = neuron.m_current/(1 - (@beta1)**@time_step)
          # v_hat = neuron.v_current/(1 - (@beta2)**@time_step)
          # neuron.bias -= (@alpha*m_hat)/(v_hat**0.5 + @epsilon)

          # neuron.m_prev = neuron.m_current
          # neuron.v_prev = neuron.v_current
        end
      end
    end

    def inspect(what : String)
      case what
      when "weights"
        @filters.each_with_index do |filter, i|
          puts "---"
          puts "Filter #{i}, weights:"
          filter.synapses.each_with_index do |channel, j|
            puts "Channel: #{j}"
            channel.each { |row| puts "#{row.map { |syn| syn.weight.round(4) }}" }
          end
        end
      when "bias"
        @filters.each_with_index { |filter, i| puts "Filter #{i}, bias:#{filter.bias.round(4)}" }
      when "activations"
        @filters.each_with_index do |filter, f|
          puts "---"
          puts "Filter: #{f}, neuron activations are:"
          filter.neurons.each do |row|
            puts "#{row.map { |n| n.activation.round(4) }}"
          end
        end
      when "gradients"
        @filters.each_with_index do |filter, f|
          puts "---"
          puts "Filter: #{f}, neuron gradients are:"
          filter.neurons.each do |row|
            puts "#{row.map { |n| n.gradient.round(4) }}"
          end
        end
      end
      puts "------------------------------------------------"
    end
  end
end
