require "logger"

module SHAInet
  class FullyConnectedLayer
    getter filters : Array(Filter), prev_layer : CNNLayer | ConvLayer
    getter output : Array(Float64), :all_neurons, :all_synapses

    def initialize(@prev_layer : CNNLayer | ConvLayer,
                   l_size : Int32,
                   @activation_function : Proc(GenNum, Array(Float64)) = SHAInet.none,
                   @logger : Logger = Logger.new(STDOUT))
      #
      # since this is similar to a classic layer, we store all neurons in a single array
      filters = height = 1
      width = l_size

      @filters = Array(Filter).new(filters) { Filter.new([width, height, filters]) }

      @output = Array(Float64).new(l_size) { 0.0 }
      @all_neurons = Array(Neuron).new
      @all_synapses = Array(Synapse).new

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

    def error_prop
      @prev_layer.filters.size.times do |filter|
        @prev_layer.filters[filter].neurons.size.times do |row|
          @prev_layer.filters[filter].neurons[row].size.times do |neuron|
            @prev_layer.filters[filter].neurons[row][neuron].hidden_error_prop
          end
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
