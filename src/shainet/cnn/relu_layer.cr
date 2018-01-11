require "logger"

module SHAInet
  class ReluLayer
    getter filters : Array(Array(Array(Array(Neuron)))) | Array(Filter), l_relu_slope : Float64, prev_layer : CNNLayer | ConvLayer
    property next_layer : CNNLayer | ConvLayer | DummyLayer

    # Calls different activaton based on previous layer type
    def activate
      _activate(@prev_layer)
    end

    #################################################
    # # This part is for dealing with conv layers # #

    # Add slope to initialize as leaky relu
    def initialize(prev_layer : ConvLayer, @l_relu_slope : Float64 = 0.0, @logger : Logger = Logger.new(STDOUT))
      # In conv layers channels is always 1, but have may multiple filters
      channels = 1
      filters = prev_layer.filters.size

      # neurons are contained in Layer class
      width = height = prev_layer.filters.first.neurons.size # Assumes row == height
      @prev_layer = prev_layer

      # Channel data is stored within the filters array
      # This is because after convolution each filter has different feature maps
      @filters = Array(Array(Array(Array(Neuron)))).new(filters) {
        Array(Array(Array(Neuron))).new(channels) {
          Array(Array(Neuron)).new(height) {
            Array(Neuron).new(width) { Neuron.new("memory") }
          }
        }
      }

      @next_layer = DummyLayer.new
      @prev_layer.next_layer = self
    end

    def _activate(prev_layer : ConvLayer)
      input_data = prev_layer.filters

      # In conv layers channels is always 1, but have may multiple filters
      input_data.size.times do |filter|
        input_data[filter].neurons.size.times do |row|
          input_data[filter].neurons[row].size.times do |col|
            if @l_relu_slope == 0.0
              @filters[filter][0][row][col].activation = SHAInet._relu(input_data[filter].neurons[row][col].activation)
            else
              @filters[filter][0][row][col].activation = SHAInet._l_relu(input_data[filter].neurons[row][col].activation)
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
      width = height = prev_layer.filters.first.first.size # Assumes row == height
      @prev_layer = prev_layer

      # Channel data is stored within the filters array
      # This is because after convolution each filter has different feature maps
      @filters = Array(Array(Array(Array(Neuron)))).new(filters) {
        Array(Array(Array(Neuron))).new(channels) {
          Array(Array(Neuron)).new(height) {
            Array(Neuron).new(width) { Neuron.new("memory") }
          }
        }
      }

      @next_layer = DummyLayer.new
      @prev_layer.next_layer = self
    end

    def _activate(prev_layer : CNNLayer)
      input_data = prev_layer.filters

      input_data.size.times do |filter|
        input_data[filter].size.times do |channel|
          input_data[filter][channel].size.times do |row|
            input_data[filter][channel][row].size.times do |col|
              if @l_relu_slope == 0.0
                @filters[filter][channel][row][col].activation = SHAInet._relu(input_data[filter][channel][row][col].activation)
              else
                @filters[filter][channel][row][col].activation = SHAInet._l_relu(input_data[filter][channel][row][col].activation)
              end
            end
          end
        end
      end
    end

    def error_prop
      _error_prop(@next_layer)
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

    def _error_prop(next_layer : DummyLayer)
      # Do nothing because this is the last layer in the network
    end

    def inspect(what : String)
      puts "ReLu layer:"
      case what
      when "weights"
        puts "ReLu layer has no wights"
      when "bias"
        puts "ReLu layer has no bias"
      when "activations"
        @filters.each_with_index do |filter, f|
          puts "Filter: #{f}"
          filter.each_with_index do |channel, ch|
            puts "Channel: #{ch}, neuron activations are:"
            channel.each do |row|
              puts "#{row.map { |n| n.activation }}"
            end
          end
        end
      end
      puts "------------"
    end
  end
end
