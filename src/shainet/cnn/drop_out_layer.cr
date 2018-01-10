require "logger"

module SHAInet
  class DropoutLayer
    getter input_layer : CNNLayer | ConvLayer, filters : Array(Array(Array(Array(Neuron)))), drop_percent : Int32

    # Calls different activaton based on previous layer type
    def activate
      _activate(@input_layer)
    end

    #################################################
    # # This part is for dealing with conv layers # #

    # Drop percent is an Int, i.e 5 is 5%
    def initialize(@input_layer : ConvLayer, @drop_percent : Int32 = 5, @logger : Logger = Logger.new(STDOUT))
      raise CNNInitializationError.new("Drop percent must be Int of 0-100") unless (0..100).includes?(@drop_percent)

      filters = @input_layer.filters.size
      channels = 1
      width = height = @input_layer.filters.neurons.size # Assumes row == height

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

    # Randomly select and deactivate a percentage of the neurons from the previous layer
    def _activate(input_layer : ConvLayer)
      input_data = input_layer.filters
      input_data.size.times do |filter|
        input_data[filter].neurons.size.times do |row|
          input_data[filter].neurons[row].size.times do |col|
            x = rand(0..100)
            if x <= @drop_percent
              @filters[filter][0][row][col].activation = 0.0
            end
          end
        end
      end
    end

    #######################################################################
    # # This part is for dealing with all layers other than conv layers # #

    # Drop percent is an Int, i.e 5 is 5%
    def initialize(@input_layer : CNNLayer, @drop_percent : Int32 = 5, @logger : Logger = Logger.new(STDOUT))
      raise CNNInitializationError.new("Drop percent must be Int of 0-100") unless (0..100).includes?(@drop_percent)

      filters = 1
      channels = input_layer.filters.first.size
      width = height = input_layer.filters.first.first.size # Assumes row == height

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
      input_data.first.size.times do |channel|
        input_data.first[channel].size.times do |row|
          input_data.first[channel][row].size.times do |col|
            x = rand(0..100)
            if x <= @drop_percent
              @filters[0][channel][row][col].activation = 0.0
            end
          end
        end
      end
    end

    def inspect(what : String)
      case what
      when "weights"
        puts "input layer has no wights"
      when "bias"
      when "neurons"
      end
    end
  end
end
