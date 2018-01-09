require "logger"

module SHAInet
  class MaxPoolLayer
    getter input_data : Array(Array(Array(Array(Neuron)))) | Array(Filter), filters : Array(Array(Array(Array(Neuron)))), pool : Int32
    @layer_type : ConvLayer.class | CNNLayerClass

    # Calls different activaton based on previous layer type
    def activate
      _activate(@layer_type)
    end

    # # This part is for dealing with conv layers # #

    # Pool refers to the one dimention for window of pixels, i.e. 2 is a window of 2x2 pixels
    def initialize(input_layer : ConvLayer, @pool : Int32, @stride : Int32, @logger : Logger = Logger.new(STDOUT))
      prev_w = input_layer.filters.first.neurons.size # Assumes row == height
      new_w = (prev_w - @pool)/@stride + 1
      raise CNNInitializationError.new("Max pool layer parameters are incorrect") if new_w.class != Int32

      @input_data = input_layer.filters
      @layer_type = input_layer.class

      filters = input_layer.filters.size
      channels = 1
      width = height = new_w # Assumes row == height

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

    def _activate(_l : ConvLayer)
      @input_data.size.times do |filter|
        input_x = input_y = output_x = output_y = 0

        # Zoom in on a small window out of the data matrix and update
        until input_y == (@input_data[filter].neurons.size - @pool - 1)
          until input_x == (@input_data[filter].neurons[input_y].size - @pool - 1)
            window = @input_data[filter].neurons[input_y..(input_y + @pool - 1)].map { |m| m[input_x..(input_x + @pool - 1)] }
            @filters[filter][0][output_y][output_x].activation = window.flatten.max
            input_x += @pool
            output_x += 1
          end
          input_x = output_x = 0
          input_y += @pool
          output_y += 1
        end
      end
    end

    # # This part is for dealing with all layers other than conv layers # #

    def initialize(input_layer : CNNLayer, @pool : Int32, @stride : Int32, @logger : Logger = Logger.new(STDOUT))
      prev_w = input_layer.filters.first.size # Assumes row == height
      new_w = (prev_w - @pool)/@stride + 1
      raise CNNInitializationError.new("Max pool layer parameters are incorrect") if new_w.class != Int32

      @input_data = input_layer.filters
      @layer_type = input_layer.class

      filters = 1
      channels = input_layer.filters.first.size
      width = height = new_w # Assumes row == height

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

    def _activate(_l : CNNLayer)
      @input_data.first.size.times do |channel|
        input_x = input_y = output_x = output_y = 0

        # Zoom in on a small window out of the data matrix and update
        until input_y == (@input_data.first[channel].size - @pool - 1)
          until input_x == (@input_data.first[channel][input_y].size - @pool - 1)
            window = @input_data.first[channel][input_y..(input_y + @pool - 1)].map { |m| m[input_x..(input_x + @pool - 1)] }
            @filters[0][channel][output_y][output_x].activation = window.flatten.max
            input_x += @pool
            output_x += 1
          end
          input_x = output_x = 0
          input_y += @pool
          output_y += 1
        end
      end
    end

    def inspect(what : String)
      puts "Maxpool layer:"
      case what
      when "weights"
        puts "Maxpool layer has no wights"
      when "bias"
        puts "Maxpool layer has no bias"
      when "activations"
        @filters.each_with_index do |filter, f|
          puts "Filter: #{f}"
          filter.each_with_index do |channel, ch|
            puts "Channel: #{ch}, neuron activationions are:"
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
