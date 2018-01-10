require "logger"

module SHAInet
  class ReluLayer
    getter input_layer : CNNLayer | ConvLayer, filters : Array(Array(Array(Array(Neuron)))), l_relu_slope : Float64

    # Calls different activaton based on previous layer type
    def activate
      _activate(@input_layer)
    end

    #################################################
    # # This part is for dealing with conv layers # #

    # Add slope to initialize as leaky relu
    def initialize(input_layer : ConvLayer, @l_relu_slope : Float64 = 0.0, @logger : Logger = Logger.new(STDOUT))
      # In conv layers channels is always 1, but have may multiple filters
      channels = 1
      filters = input_layer.filters.size

      # neurons are contained in Layer class
      width = height = input_layer.filters.first.neurons.size # Assumes row == height
      @input_layer = input_layer

      # Channel data is stored within the filters array
      # This is because after convolution each filter has different feature maps
      @filters = Array(Array(Array(Array(Neuron)))).new(filters) {
        Array(Array(Array(Neuron))).new(channels) {
          Array(Array(Neuron)).new(height) {
            Array(Neuron).new(width) { Neuron.new("memory") }
          }
        }
      }
    end

    def _activate(input_layer : ConvLayer)
      input_data = input_layer.filters

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

    def initialize(input_layer : CNNLayer, @l_relu_slope : Float64 = 0.0, @logger : Logger = Logger.new(STDOUT))
      # In other layers filters is always 1, but may have multiple channels
      channels = input_layer.filters.first.size
      filters = 1
      # Neurons are contained in Multi-Array
      width = height = input_layer.filters.first.first.size # Assumes row == height
      @input_layer = input_layer

      # Channel data is stored within the filters array
      # This is because after convolution each filter has different feature maps
      @filters = Array(Array(Array(Array(Neuron)))).new(filters) {
        Array(Array(Array(Neuron))).new(channels) {
          Array(Array(Neuron)).new(height) {
            Array(Neuron).new(width) { Neuron.new("memory") }
          }
        }
      }
    end

    def _activate(input_layer : CNNLayer)
      input_data = input_layer.filters

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
