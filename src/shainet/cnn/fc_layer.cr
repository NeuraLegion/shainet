require "logger"

module SHAInet
  class FullyConnectedLayer
    getter filters : Array(Filter), prev_layer : CNNLayer | ConvLayer
    getter output : Array(Float64), :all_neurons, :all_synapses

    def initialize(@master_network : CNN,
                   @prev_layer : CNNLayer | ConvLayer,
                   l_size : Int32,
                   @activation_function : ActivationFunction = SHAInet.none,
                   @logger : Logger = Logger.new(STDOUT))
      #
      # since this is similar to a classic layer, we store all neurons in a single array
      filters = height = 1
      width = l_size

      @filters = Array(Filter).new(filters) { Filter.new([width, height, filters]) }

      @output = Array(Float64).new(l_size) { 0.0 }
      @all_neurons = Array(Neuron).new
      @all_synapses = Array(Synapse).new

      @w_gradient = Array(Float64).new # Needed for batch train
      @b_gradient = Array(Float64).new # Needed for batch train

      # Connect the last layer to the neurons of this layer (fully connect)
      @filters.first.neurons.first.each do |target_neuron|
        @prev_layer.filters.size.times do |filter|
          @prev_layer.filters[filter].neurons.size.times do |row|
            @prev_layer.filters[filter].neurons[row].each do |source_neuron|
              synapse = Synapse.new(source_neuron, target_neuron)
              source_neuron.synapses_out << synapse
              target_neuron.synapses_in << synapse
              @all_neurons << target_neuron

              @all_synapses << synapse
            end
          end
        end
      end
    end

    def activate
      @filters.first.neurons.first.each_with_index do |neuron, i|
        neuron.activate(@activation_function)
        @output[i] = neuron.activation
      end
    end

    def error_prop(batch : Bool = false)
      @prev_layer.filters.size.times do |filter|
        @prev_layer.filters[filter].neurons.size.times do |row|
          @prev_layer.filters[filter].neurons[row].size.times do |neuron|
            target_neuron = @prev_layer.filters[filter].neurons[row][neuron]
            target_neuron.hidden_error_prop
            if batch == true
              target_neuron.gradient_sum += target_neuron.gradient
              target_neuron.synapses_out.each do |synapse|
                synapse.gradient_sum += synapse.source_neuron.activation*synapse.dest_neuron.gradient
              end
            else
              target_neuron.synapses_out.each do |synapse|
                synapse.gradient = synapse.source_neuron.activation*synapse.dest_neuron.gradient
              end
            end
          end
        end
      end
    end

    def update_wb(learn_type : Symbol | String, batch : Bool = false)
      # Update all weights of the layer
      @all_synapses.each do |synapse|
        if batch == true
          synapse.gradient = synapse.gradient_sum
          synapse.gradient_sum = Float64.new(0) # Reset the gradient for next batch
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

      # Update all biases of the layer
      @all_neurons.each do |neuron|
        if batch == true
          neuron.gradient = neuron.gradient_sum
          neuron.gradient_sum = Float64.new(0)
        end

        case learn_type.to_s
        # Update biases based on the gradients and delta rule (including momentum)
        when "sgdm"
          delta_bias = (-1)*@master_network.learning_rate*(neuron.gradient) + @master_network.momentum*(neuron.bias - neuron.prev_bias)
          neuron.bias += delta_bias
          neuron.prev_bias = neuron.bias

          # Update weights based on Resilient backpropogation (Rprop), using the improved varient iRprop+
        when "rprop"
          if neuron.prev_gradient*neuron.gradient > 0
            delta = [@master_network.etah_plus*neuron.prev_delta, @master_network.delta_max].min
            delta_bias = (-1)*SHAInet.sign(neuron.gradient)*delta

            neuron.bias += delta_bias
            neuron.prev_bias = neuron.bias
            neuron.prev_delta = delta
            neuron.prev_delta_b = delta_bias
          elsif neuron.prev_gradient*neuron.gradient < 0.0
            delta = [@master_network.etah_minus*neuron.prev_delta, @master_network.delta_min].max

            neuron.bias -= neuron.prev_delta_b if @master_network.mean_error >= @master_network.prev_mean_error

            neuron.prev_gradient = 0.0
            neuron.prev_delta = delta
          elsif neuron.prev_gradient*neuron.gradient == 0.0
            delta_bias = (-1)*SHAInet.sign(neuron.gradient)*@master_network.delta_min*neuron.prev_delta

            neuron.bias += delta_bias
            neuron.prev_delta = @master_network.delta_min
            neuron.prev_delta_b = delta_bias
          end
          #
          # Update weights based on Adaptive moment estimation (Adam)
        when "adam"
          neuron.m_current = @master_network.beta1*neuron.m_prev + (1 - @master_network.beta1)*neuron.gradient
          neuron.v_current = @master_network.beta2*neuron.v_prev + (1 - @master_network.beta2)*(neuron.gradient)**2

          m_hat = neuron.m_current/(1 - (@master_network.beta1)**@master_network.time_step)
          v_hat = neuron.v_current/(1 - (@master_network.beta2)**@master_network.time_step)
          neuron.bias -= (@master_network.alpha*m_hat)/(v_hat**0.5 + @master_network.epsilon)

          neuron.m_prev = neuron.m_current
          neuron.v_prev = neuron.v_current
        end
      end
    end

    def inspect(what : String)
      case what
      when "weights"
        @filters.first.neurons.first.each_with_index do |neuron, i|
          puts "Neuron: #{i}, incoming weights:"
          puts "#{neuron.synapses_in.map { |synapse| synapse.weight.round(4) }}"
        end
      when "bias"
        @filters.first.neurons.first.each_with_index { |neuron, i| puts "Neuron: #{i}, bias: #{neuron.bias.round(4)}" }
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
