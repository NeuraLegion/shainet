require "logger"

module SHAInet
  class ReluLayer
    getter filters : Array(Array(Array(Array(Neuron)))), l_relu_slope : Float64, prev_layer : CNNLayer | ConvLayer
    property next_layer : CNNLayer | ConvLayer | DummyLayer

    # Calls different activaton based on previous layer type
    def activate
      _activate(@prev_layer)
    end

    #################################################
    # # This part is for dealing with conv layers # #

    # Add slope to initialize as leaky relu
    def initialize(prev_layer : ConvLayer, @l_relu_slope : Float64, @logger : Logger = Logger.new(STDOUT))
      # In conv layers channels is always 1, but have may multiple filters
      filters = prev_layer.filters.size
      channels = 1
      # neurons are contained in Layer class
      width = height = prev_layer.filters.first.neurons.size # Assumes row == height

      # Channel data is stored within the filters array
      # This is because after convolution each filter has different feature maps
      @filters = Array(Array(Array(Array(Neuron)))).new(filters) {
        Array(Array(Array(Neuron))).new(channels) {
          Array(Array(Neuron)).new(height) {
            Array(Neuron).new(width) { Neuron.new("memory") }
          }
        }
      }

      @prev_layer = prev_layer
      @next_layer = DummyLayer.new
      @prev_layer.next_layer = self
    end

    def _activate(prev_layer : ConvLayer)
      @filters.size.times do |filter|
        @filters[filter][0].size.times do |row| # Conv layers may have multiple filters, but only one channel
          @filters[filter][0][row].size.times do |neuron|
            if @l_relu_slope == 0.0
              @filters[filter][0][row][neuron].activation = SHAInet._relu(prev_layer.filters[filter].neurons[row][neuron].activation)
            else
              @filters[filter][0][row][neuron].activation = SHAInet._l_relu(prev_layer.filters[filter].neurons[row][neuron].activation)
            end
          end
        end
      end
    end

    #######################################################################
    # # This part is for dealing with all layers other than conv layers # #

    def initialize(prev_layer : CNNLayer, @l_relu_slope : Float64 = 0.0, @logger : Logger = Logger.new(STDOUT))
      # In other layers filters is always 1, but may have multiple channels
      channels = prev_layer.filters.first.size
      filters = 1
      # Neurons are contained in Multi-Array
      height = prev_layer.filters.first.first.size
      width = prev_layer.filters.first.first.first.size

      # Channel data is stored within the filters array
      # This is because after convolution each filter has different feature maps
      @filters = Array(Array(Array(Array(Neuron)))).new(filters) {
        Array(Array(Array(Neuron))).new(channels) {
          Array(Array(Neuron)).new(height) {
            Array(Neuron).new(width) { Neuron.new("memory") }
          }
        }
      }
      # @filters = prev_layer.filters.clone

      @prev_layer = prev_layer
      @next_layer = DummyLayer.new
      @prev_layer.next_layer = self
    end

    def _activate(prev_layer : CNNLayer)
      @filters.size.times do |filter|
        @filters[filter].size.times do |channel|
          @filters[filter][channel].size.times do |row| # Conv layers may have multiple filters, but only one channel
            @filters[filter][channel][row].size.times do |neuron|
              if @l_relu_slope == 0.0
                @filters[filter][channel][row][neuron].activation = SHAInet._relu(prev_layer.filters[filter][channel][row][neuron].activation)
              else
                @filters[filter][channel][row][neuron].activation = SHAInet._l_relu(prev_layer.filters[filter][channel][row][neuron].activation)
              end
            end
          end
        end
      end
    end

    def error_prop
      _error_prop(@next_layer)
    end

    def _error_prop(next_layer : ReluLayer | DropoutLayer)
      @filters.size.times do |filter|
        @filters[filter].size.times do |channel|
          @filters[filter][channel].size.times do |row|
            @filters[filter][channel][row].size.times do |neuron|
              @filters[filter][channel][row][neuron].gradient = next_layer.filters[filter][channel][row][neuron].gradient
            end
          end
        end
      end
    end

    def _error_prop(next_layer : MaxPoolLayer)
      @filters.size.times do |filter|
        @filters[filter].size.times do |channel|
          input_x = input_y = output_x = output_y = 0

          while input_y < (@filters[filter][channel].size - next_layer.pool + next_layer.stride)   # Break out of y
            while input_x < (@filters[filter][channel].size - next_layer.pool + next_layer.stride) # Break out of x (assumes x = y)
              pool_neuron = next_layer.filters[filter][channel][output_y][output_x]

              # Only propagate error to the neurons that were chosen during the max pool
              @filters[filter][channel][input_y..(input_y + next_layer.pool - 1)].each do |row|
                row[input_x..(input_x + next_layer.pool - 1)].each do |neuron|
                  if neuron.activation == pool_neuron.activation
                    neuron.gradient = pool_neuron.gradient
                  end
                end
              end

              input_x += next_layer.stride
              output_x += 1
            end
            input_x = output_x = 0
            input_y += next_layer.stride
            output_y += 1
          end
        end
      end
    end

    def _error_prop(next_layer : FullyConnectedLayer)
      @filters.each do |filter|
        filter.each do |channel|
          channel.each do |row|
            row.each { |neuron| neuron.hidden_error_prop }
          end
        end
      end
    end

    def _error_prop(next_layer : InputLayer | SoftmaxLayer | DummyLayer | ConvLayer)
      # Do nothing
    end

    def inspect(what : String)
      puts "ReLu layer:"
      case what
      when "weights"
        puts "ReLu layer has no weights"
      when "bias"
        puts "ReLu layer has no bias"
      when "activations"
        @filters.each_with_index do |filter, f|
          puts "Filter: #{f}"
          filter.each_with_index do |channel, ch|
            puts "Channel: #{ch}, neuron activations are:"
            channel.each do |row|
              puts "#{row.map { |n| n.activation.round(4) }}"
            end
          end
        end
      end
      puts "------------"
    end
  end
end
