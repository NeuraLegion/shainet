require "logger"

module SHAInet
  class InputLayer
    getter :filters, :output
    property next_layer : CNNLayer | ConvLayer | DummyLayer

    def initialize(input_volume : Array(Int32), @logger : Logger = Logger.new(STDOUT))
      unless input_volume.size == 3
        raise CNNInitializationError.new("Input volume must be an array of Int32: [width, height, channels].")
      end

      unless input_volume[0] == input_volume[1]
        raise CNNInitializationError.new("Width and height of input must be of the same size.")
      end

      filters = 1 # In this case there is only one filter, since it is the input layer
      channels = input_volume[2]
      width = input_volume[0]
      height = input_volume[1]

      # Channel data is stored within the filters array, this is needed for smooth work with all other layers.
      @filters = Array(Array(Array(Array(Neuron)))).new(filters) {
        Array(Array(Array(Neuron))).new(channels) {
          Array(Array(Neuron)).new(height) {
            Array(Neuron).new(width) { Neuron.new("memory") }
          }
        }
      }

      @next_layer = DummyLayer.new
    end

    def activate(input_data : Array(Array(Array(GenNum))))
      # Input the data into the first layer
      input_data.size.times do |channel|
        input_data[channel].size.times do |row|
          input_data[channel][row].size.times do |col|
            @filters.first[channel][row][col].activation = input_data[channel][row][col].to_f64
          end
        end
      end
    end

    def error_prop
      _error_prop(@next_layer)
    end

    def _error_prop(next_layer : ConvLayer)
      # @filters.each do |filter|
      #   filter.propagate_backward(next_layer)
      # end
    end

    def _error_prop(next_layer : CNNLayer | DummyLayer)
      # Do nothing
    end

    def propagate_backward(next_layer : ConvLayer)
      padded_data = SHAInet::Filter._pad(@channels, next_layer.padding) # Array of all channels or all filters

      # Starting locations
      input_x = input_y = output_x = output_y = 0

      # Update the gradients of all neurons in current layer and weight gradients for the filters of the next layer
      next_layer.filters.size.times do |filter|
        # Takes a small window from the input data (Channel/Filter x Width x Height) to preform feed forward
        # Slides the window over the input data volume and updates each neuron of the filter
        # The window depth is the number of all channels/filters (depending on previous layer)
        while input_y < (padded_data.first.size - @window_size + @stride)         # Break out of y
          while input_x < (padded_data.first.first.size - @window_size + @stride) # Break out of x
            window = padded_data.map { |self_filter| self_filter[input_y..(input_y + @window_size - 1)].map { |row| row[input_x..(input_x + @window_size - 1)] } }
            source_neuron = next_layer.filters[filter].neurons[output_y][output_x]

            # update the weighted error for the entire window
            synapses = next_layer.filters[filter].synapses
            # input_sum = Float64.new(0)
            synapses.size.times do |channel|
              synapses[channel].size.times do |row|
                synapses[channel][row].size.times do |col| # Synapses are CnnSynpase in this case
                # Save the error sum for updating the weights later
                  target_neuron = @channels[channel][row][col]
                  synapses[channel][row][col].gradient_sum += source_neuron.gradient*target_neuron.activation
                end
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

    def inspect(what : String)
      puts "Input layer:"
      case what
      when "weights"
        puts "input layer has no weights"
      when "bias"
        puts "input layer has no bias"
      when "activations"
        @filters.first.each_with_index do |channel, ch|
          puts "Channel: #{ch}, neuron activations are:"
          channel.each do |row|
            puts "#{row.map { |n| n.activation.round(4) }}"
          end
        end
      end
      puts "------------"
    end
  end
end
