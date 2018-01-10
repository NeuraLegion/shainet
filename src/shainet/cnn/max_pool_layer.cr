require "logger"

module SHAInet
  class MaxPoolLayer
    getter input_layer : CNNLayer | ConvLayer, filters : Array(Array(Array(Array(Neuron)))), pool : Int32

    # Calls different activaton based on previous layer type
    def activate
      _activate(@input_layer)
    end

    #################################################
    # # This part is for dealing with conv layers # #

    # Pool refers to the one dimention for window of pixels, i.e. 2 is a window of 2x2 pixels
    def initialize(@input_layer : ConvLayer, @pool : Int32, @stride : Int32, @logger : Logger = Logger.new(STDOUT))
      prev_w = input_layer.filters.first.neurons.size # Assumes row == height
      new_w = ((prev_w.to_f64 - @pool.to_f64)/@stride.to_f64 + 1).to_f64
      raise CNNInitializationError.new("Max pool layer parameters are incorrect") unless new_w.to_i == new_w

      filters = input_layer.filters.size
      channels = 1
      width = height = new_w.to_i # Assumes row == height

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

      input_data.each_with_index do |_fi, self_filter|
        input_x = input_y = output_x = output_y = 0

        input_data.each_with_index do |_f, filter|
          # Zoom in on a small window out of the data matrix and update
          input_x = input_y = output_x = output_y = 0

          while input_y < (input_data[filter].neurons.size - @pool + @stride)   # Break out of y
            while input_x < (input_data[filter].neurons.size - @pool + @stride) # Break out of x (assumes x = y)
              window = input_data[filter].neurons[input_y..(input_y + @pool - 1)].map { |row| row[input_x..(input_x + @pool - 1)].map { |neuron| neuron.activation } }

              @filters[self_filter][0][output_y][output_x].activation = window.flatten.max
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

    #######################################################################
    # # This part is for dealing with all layers other than conv layers # #

    def initialize(@input_layer : CNNLayer, @pool : Int32, @stride : Int32, @logger : Logger = Logger.new(STDOUT))
      prev_w = input_layer.filters.first.first.size # Assumes row == height
      new_w = ((prev_w.to_f64 - @pool.to_f64)/@stride.to_f64 + 1).to_f64
      puts "new width: #{new_w}"
      raise CNNInitializationError.new("Max pool layer parameters are incorrect") unless new_w.to_i == new_w

      filters = 1
      channels = input_layer.filters.first.size
      width = height = new_w.to_i # Assumes row == height

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
      input_data.first.each_with_index do |_ch, channel|
        # Zoom in on a small window out of the data matrix and update
        input_x = input_y = output_x = output_y = 0
        while input_y < (input_data[0][channel].size - @pool + @stride)   # Break out of y
          while input_x < (input_data[0][channel].size - @pool + @stride) # Break out of x (assumes x = y)
            window = input_data[0][channel][input_y..(input_y + @pool - 1)].map { |row| row[input_x..(input_x + @pool - 1)].map { |neuron| neuron.activation } }

            @filters[0][channel][output_y][output_x].activation = window.flatten.max
            input_x += @stride
            output_x += 1
          end
          input_x = output_x = 0
          input_y += @stride
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
