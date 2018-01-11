require "logger"

module SHAInet
  class FullyConnectedLayer
    getter filters : Array(Array(Array(Array(Neuron)))), prev_layer : CNNLayer | ConvLayer
    property next_layer : CNNLayer | ConvLayer | DummyLayer
    getter output : Array(Float64), :all_neurons, :all_synapses
    @softmax : Bool

    #################################################
    # # This part is for dealing with conv layers # #

    def initialize(prev_layer : ConvLayer,
                   l_size : Int32,
                   @softmax : Bool = false,
                   @activation_function : Proc(GenNum, Array(Float64)) = SHAInet.sigmoid,
                   @logger : Logger = Logger.new(STDOUT))
      #

      filters = channels = height = 1
      width = l_size # since this is similar to a classic layer, we store all neurons in a single array

      # Channel data is stored within the filters array, this is needed for smooth work with all other layers.
      @filters = Array(Array(Array(Array(Neuron)))).new(filters) {
        Array(Array(Array(Neuron))).new(channels) {
          Array(Array(Neuron)).new(height) {
            Array(Neuron).new(width) { Neuron.new("memory") }
          }
        }
      }

      @output = Array(Float64).new(l_size) { 0.0 }
      @all_neurons = Array(Neuron).new
      @all_synapses = Array(Synapse).new

      # Connect the last layer to the output layer (fully connect)
      @filters.first.first.first.each do |neuron2| # Target neuron
        prev_layer.filters.size.times do |filter|
          prev_layer.filters[filter].neurons.size.times do |row|
            prev_layer.filters[filter].neurons[row].each do |neuron1| # Source neuron
              synapse = Synapse.new(neuron1, neuron2)
              neuron1.synapses_out << synapse
              neuron2.synapses_in << synapse
              @all_neurons << neuron2
              @all_synapses << synapse
            end
          end
        end
      end
      @prev_layer = prev_layer
      @next_layer = DummyLayer.new
      prev_layer.next_layer = self
    end

    #######################################################################
    # # This part is for dealing with all layers other than conv layers # #

    def initialize(prev_layer : CNNLayer,
                   l_size : Int32,
                   @softmax : Bool = false,
                   @activation_function : Proc(GenNum, Array(Float64)) = SHAInet.sigmoid,
                   @logger : Logger = Logger.new(STDOUT))
      #

      filters = channels = height = 1
      width = l_size # since this is similar to a classic layer, we store all neurons in a single array

      # Channel data is stored within the filters array, this is needed for smooth work with all other layers.
      @filters = Array(Array(Array(Array(Neuron)))).new(filters) {
        Array(Array(Array(Neuron))).new(channels) {
          Array(Array(Neuron)).new(height) {
            Array(Neuron).new(width) { Neuron.new("memory") }
          }
        }
      }

      @output = Array(Float64).new(l_size) { 0.0 }
      @all_neurons = Array(Neuron).new
      @all_synapses = Array(Synapse).new

      # Connect the last layer to the output layer (fully connect)
      @filters.first.first.first.each do |neuron2| # Target neuron
        prev_layer.filters.first.size.times do |channel|
          prev_layer.filters.first[channel].size.times do |row|
            prev_layer.filters.first[channel][row].each do |neuron1| # Source neuron
              synapse = Synapse.new(neuron1, neuron2)
              neuron1.synapses_out << synapse
              neuron2.synapses_in << synapse
              @all_neurons << neuron2
              @all_synapses << synapse
            end
          end
        end
      end
      @prev_layer = prev_layer
      @next_layer = DummyLayer.new
      prev_layer.next_layer = self
    end

    def activate
      if @softmax == true
        sf_activations = [] of Float64
        @filters.first.first.each do |row|
          activations = [] of Float64
          # Calculate the softmax values based on entire row
          row.each do |neuron|
            neuron.activate(@activation_function = SHAInet.none)
            activations << neuron.activation
            sf_activations = SHAInet.softmax(activations)
          end
          # Update the neuron activations to fit the softmax values
          row.each_with_index do |neuron, i|
            neuron.activation = sf_activations[i]
          end
        end
        @output = sf_activations
      else
        @filters.first.first.first.each_with_index do |neuron, i|
          neuron.activate(@activation_function)
          @output[i] = neuron.activation
        end
      end
    end

    def error_prop
      _error_prop(@next_layer)
    end

    def _error_prop(next_layer : ReluLayer | DropoutLayer)
      @filters.each_with_index do |filter, fi|
        filter.each_with_index do |channel, ch|
          channel.each_with_index do |row, r|
            row.each_with_index do |neuron, n|
              neuron.gradient = next_layer.filters[fi][ch][r][n].gradient
            end
          end
        end
      end
    end

    def _error_prop(next_layer : MaxPoolLayer)
      @filters.each_with_index do |_f, filter|
        _f.each_with_index do |_ch, channel|
          input_x = input_y = output_x = output_y = 0

          while input_y < (@filters[filter][channel].size - @pool + @stride)   # Break out of y
            while input_x < (@filters[filter][channel].size - @pool + @stride) # Break out of x (assumes x = y)
              pool_neuron = next_layer.filters[filter][channel][output_y][output_x]

              # Only propagate error to the neurons that were chosen during the max pool
              @filters[filter][channel][input_y..(input_y + @pool - 1)].each do |row|
                row[input_x..(input_x + @pool - 1)].each do |neuron|
                  if neuron.activation == pool_neuron.activation
                    neuron.gradient = pool_neuron.gradient
                  end
                end
              end

              input_x += @stride
              output_x += 1
            end
            input_x = output_x = 0
            input_y += @stride
            output_y += 1
          end
        end
      end
    end

    def _error_prop(next_layer : FullyConnectedLayer)
      @filters.first.first.first.each { |neuron| neuron.hidden_error_prop }
    end

    def _error_prop(next_layer : DummyLayer)
      # Do nothing because this is the last layer in the network
    end

    def inspect(what : String)
      puts "Fully_connected layer:"
      case what
      when "weights"
        @filters.first.first.first.each_with_index do |neuron, i|
          puts "Neuron: #{i}, incoming weights:"
          puts "#{neuron.synapses_in.map { |synapse| synapse.weight }}"
        end
      when "bias"
        @filters.first.first.first.each_with_index { |neuron, i| puts "Neuron: #{i}, bias: #{neuron.bias}" }
      when "activations"
        @filters.first.first.each { |row| puts "#{row.map { |n| n.activation }}" }
      end
      puts "------------"
    end
  end
end
