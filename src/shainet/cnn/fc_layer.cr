require "logger"

module SHAInet
  class FullyConnectedLayer
    getter input_data : Array(Array(Array(Array(Neuron)))) | Array(Filter), filters : Array(Array(Array(Array(Neuron)))), :all_neurons, :all_synapses

    # # This part is for dealing with conv layers # #

    def initialize(input_layer : ConvLayer,
                   l_size : Int32,
                   @activation_function : Proc(GenNum, Array(Float64)) = SHAInet.sigmoid,
                   @logger : Logger = Logger.new(STDOUT))
      #

      filters = 1
      channels = 1
      height = 1
      width = l_size # since this is similar to a classic layer, we store all neurons in a single array

      # Channel data is stored within the filters array, this is needed for smooth work with all other layers.
      @filters = Array(Array(Array(Array(Neuron)))).new(filters) {
        Array(Array(Array(Neuron))).new(channels) {
          Array(Array(Neuron)).new(height) {
            Array(Neuron).new(width) { Neuron.new("memory") }
          }
        }
      }

      @all_neurons = Array(Neuron).new
      @all_synapses = Array(Synapse).new

      # Connect the last layer to the output layer (fully connect)
      @filters.first.first.first.each do |neuron2| # Target neuron
        input_layer.filters.size.times do |filter|
          input_layer.filters[filter].neurons.size.times do |row|
            input_layer.filters[filter].neurons[row].each do |neuron1| # Source neuron
              synapse = Synapse.new(neuron1, neuron2)
              neuron1.synapses_out << synapse
              neuron2.synapses_in << synapse
              @all_neurons << neuron2
              @all_synapses << synapse
            end
          end
        end
      end
      @input_data = input_layer.filters
    end

    # # This part is for dealing with all layers other than conv layers # #

    def initialize(input_layer : CNNLayer,
                   l_size : Int32,
                   @activation_function : Proc(GenNum, Array(Float64)) = SHAInet.sigmoid,
                   @logger : Logger = Logger.new(STDOUT))
      #

      filters = 1
      channels = 1
      height = 1
      width = l_size # since this is similar to a classic layer, we store all neurons in a single array

      # Channel data is stored within the filters array, this is needed for smooth work with all other layers.
      @filters = Array(Array(Array(Array(Neuron)))).new(filters) {
        Array(Array(Array(Neuron))).new(channels) {
          Array(Array(Neuron)).new(height) {
            Array(Neuron).new(width) { Neuron.new("memory") }
          }
        }
      }

      @all_neurons = Array(Neuron).new
      @all_synapses = Array(Synapse).new

      # Connect the last layer to the output layer (fully connect)
      @filters.first.first.first.each do |neuron2| # Target neuron
        input_layer.filters.first.size.times do |channel|
          input_layer.filters.first[channel].size.times do |row|
            input_layer.filters.first[channel][row].each do |neuron1| # Source neuron
              synapse = Synapse.new(neuron1, neuron2)
              neuron1.synapses_out << synapse
              neuron2.synapses_in << synapse
              @all_neurons << neuron2
              @all_synapses << synapse
            end
          end
        end
      end
      @input_data = input_layer.filters
    end

    def activate
      if @activation_function == SHAInet.softmax
        @filters.first.first.each do |row|
          activations = [] of Float64
          # Calculate the softmax values based on entire row
          row.each do |neuron|
            neuron.activate(@activation_function = SHAInet.none)
            activations << neuron.activation
            sf_activations = softmax(activations)
          end
          # Update the neuron activations to fit the softmax values
          row.each_with_index do |neuron, i|
            neuron.activation = sf_activations[i]
          end
        end
        return sf_activations
      else
        @filters.first.first.each { |neuron| neuron.activate(@activation_function) }
      end
    end

    def inspect(what : String)
      puts "Fully_connected layer:"
      case what
      when "weights"
        # @filters.first.first.first { |row| puts "#{row.map { |n| n.synapses_in.each { |s| s.weight } }}" }
        puts "No weights for you!"
      when "bias"
        puts "Maxpool layer has no bias"
      when "activations"
        @filters.first.first.each { |row| puts "#{row.map { |n| n.activation }}" }
      end
      puts "------------"
    end
  end
end
