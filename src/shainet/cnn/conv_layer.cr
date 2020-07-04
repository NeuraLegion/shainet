require "log"

module SHAInet
  class ConvLayer
    getter master_network : CNN, prev_layer : CNNLayer | ConvLayer, filters : Array(Filter)
    getter window_size : Int32, stride : Int32, padding : Int32, activation_function : ActivationFunction

    def initialize(@master_network : CNN,
                   @prev_layer : ConvLayer | CNNLayer,
                   filters_num : Int32 = 1,
                   @window_size : Int32 = 1,
                   @stride : Int32 = 1,
                   @padding : Int32 = 0,
                   @activation_function : ActivationFunction = SHAInet.none,
                   @log : Log = Log.new(STDOUT))
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
    end

    # Use each filter to create feature maps from the input data of the previous layer
    def activate
      @filters.each { |filter| filter.propagate_forward(@prev_layer) }
    end

    def error_prop(batch : Bool = false)
      @filters.each { |filter| filter.propagate_backward(@prev_layer, batch) }
    end

    def update_wb(learn_type : Symbol | String, batch : Bool = false)
      @filters.each do |filter|
        filter.synapses.size.times do |channel|
          filter.synapses[channel].size.times do |row|
            filter.synapses[channel][row].size.times do |col|
              synapse = filter.synapses[channel][row][col]

              if batch == true
                synapse.gradient = synapse.gradient_batch # Get current gradient
                synapse.gradient_sum = Float64.new(0)     # Reset gradient sum for next input
                synapse.gradient_batch = Float64.new(0)   # Reset gradient sum for next batch
              else
                synapse.gradient = synapse.gradient_sum # Get current gradient
                synapse.gradient_sum = Float64.new(0)   # Reset gradient sum for next input
              end

              case learn_type.to_s
              # Update weights based on the gradients and delta rule (including momentum)
              when "sgdm"
                delta_weight = (-1)*@master_network.learning_rate*synapse.gradient + @master_network.momentum*(synapse.weight - synapse.prev_weight)
                synapse.weight += delta_weight
                synapse.prev_weight = synapse.weight

                # Update weights based on Resilient backpropogation (Rprop), using the improved varient iRprop+
              when "rprop"
                if synapse.prev_gradient*synapse.gradient > 0
                  delta = [@master_network.etah_plus*synapse.prev_delta, @master_network.delta_max].min
                  delta_weight = (-1)*SHAInet.sign(synapse.gradient)*delta

                  synapse.weight += delta_weight
                  synapse.prev_weight = synapse.weight
                  synapse.prev_delta = delta
                  synapse.prev_delta_w = delta_weight
                elsif synapse.prev_gradient*synapse.gradient < 0.0
                  delta = [@master_network.etah_minus*synapse.prev_delta, @master_network.delta_min].max

                  synapse.weight -= synapse.prev_delta_w if @master_network.mean_error >= @master_network.prev_mean_error

                  synapse.prev_gradient = 0.0
                  synapse.prev_delta = delta
                elsif synapse.prev_gradient*synapse.gradient == 0.0
                  delta_weight = (-1)*SHAInet.sign(synapse.gradient)*synapse.prev_delta

                  synapse.weight += delta_weight
                  synapse.prev_delta = @master_network.delta_min
                  synapse.prev_delta_w = delta_weight
                end
                #
                # Update weights based on Adaptive moment estimation (Adam)
              when "adam"
                synapse.m_current = @master_network.beta1*synapse.m_prev + (1 - @master_network.beta1)*synapse.gradient
                synapse.v_current = @master_network.beta2*synapse.v_prev + (1 - @master_network.beta2)*(synapse.gradient)**2

                m_hat = synapse.m_current/(1 - (@master_network.beta1)**@master_network.time_step)
                v_hat = synapse.v_current/(1 - (@master_network.beta2)**@master_network.time_step)
                synapse.weight -= (@master_network.alpha*m_hat)/(v_hat**0.5 + @master_network.epsilon)

                synapse.m_prev = synapse.m_current
                synapse.v_prev = synapse.v_current
              end
            end
          end
        end

        if batch == true
          filter.bias_grad = filter.bias_grad_batch # Update biases of the layer
          filter.bias_grad_sum = Float64.new(0)     # Resest bias sum of the layer
          filter.bias_grad_batch = Float64.new(0)   # Resest bias sum of the layer
        else
          filter.bias_grad = filter.bias_grad_sum # Update biases of the layer
          filter.bias_grad_sum = Float64.new(0)   # Resest bias sum of the layer
        end

        case learn_type.to_s
        # Update biases based on the gradients and delta rule (including momentum)
        when "sgdm"
          delta_bias = (-1)*@master_network.learning_rate*(filter.bias_grad) + @master_network.momentum*(filter.bias - filter.prev_bias)
          filter.bias += delta_bias
          filter.prev_bias = filter.bias

          # Update weights based on Resilient backpropogation (Rprop), using the improved varient iRprop+
        when "rprop"
          if filter.prev_bias_grad*filter.bias_grad > 0
            delta = [@master_network.etah_plus*filter.prev_delta, @master_network.delta_max].min
            delta_bias = (-1)*SHAInet.sign(filter.bias_grad)*delta

            filter.bias += delta_bias
            filter.prev_bias = filter.bias
            filter.prev_delta = delta
            filter.prev_delta_b = delta_bias
          elsif filter.prev_bias_grad*filter.bias_grad < 0.0
            delta = [@master_network.etah_minus*filter.prev_delta, @master_network.delta_min].max

            filter.bias -= filter.prev_delta_b if @master_network.mean_error >= @master_network.prev_mean_error

            filter.prev_bias_grad = 0.0
            filter.prev_delta = delta
          elsif filter.prev_bias_grad*filter.bias_grad == 0.0
            delta_bias = (-1)*SHAInet.sign(filter.bias_grad)*@master_network.delta_min*filter.prev_delta

            filter.bias += delta_bias
            filter.prev_delta = @master_network.delta_min
            filter.prev_delta_b = delta_bias
          end
          # Update weights based on Adaptive moment estimation (Adam)
        when "adam"
          filter.m_current = @master_network.beta1*filter.m_prev + (1 - @master_network.beta1)*filter.bias_grad
          filter.v_current = @master_network.beta2*filter.v_prev + (1 - @master_network.beta2)*(filter.bias_grad)**2

          m_hat = filter.m_current/(1 - (@master_network.beta1)**@master_network.time_step)
          v_hat = filter.v_current/(1 - (@master_network.beta2)**@master_network.time_step)
          filter.bias -= (@master_network.alpha*m_hat)/(v_hat**0.5 + @master_network.epsilon)

          filter.m_prev = filter.m_current
          filter.v_prev = filter.v_current
        end
      end
    end

    def inspect(what : String)
      puts "##################################################"
      puts "ConvLayer:"
      puts "----------"
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
