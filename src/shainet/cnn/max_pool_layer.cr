require "logger"

module SHAInet
  class MaxPoolLayer
    getter filters : Array(Filter), pool : Int32, stride : Int32, prev_layer : CNNLayer | ConvLayer

    def initialize(@prev_layer : CNNLayer | ConvLayer,
                   @pool : Int32,
                   @stride : Int32,
                   @logger : Logger = Logger.new(STDOUT))
      #
      prev_w = prev_layer.filters.first.neurons.size # Assumes row == height
      new_w = ((prev_w.to_f64 - @pool.to_f64)/@stride.to_f64 + 1).to_f64
      raise CNNInitializationError.new("Max pool layer parameters are incorrect, change pool size or stride") unless new_w.to_i == new_w

      filters = @prev_layer.filters.size
      width = height = new_w.to_i # Assumes row == height

      @filters = Array(Filter).new(filters) { Filter.new([width, height, 1]) }
    end

    def activate
      _activate(@prev_layer)
    end

    def _activate(prev_layer : CNNLayer | ConvLayer)
      input_data = prev_layer.filters

      input_data.size.times do |filter|
        # Zoom in on a small window out of the data matrix and update
        input_x = input_y = output_x = output_y = 0

        while input_y < (input_data[filter].neurons.size - @pool + @stride)         # Break out of y
          while input_x < (input_data[filter].neurons.first.size - @pool + @stride) # Break out of x
            window = input_data[filter].neurons[input_y..(input_y + @pool - 1)].map { |row| row[input_x..(input_x + @pool - 1)].map { |neuron| neuron.activation } }
            @filters[filter].neurons[output_y][output_x].activation = window.flatten.max

            input_x += @stride
            output_x += 1
          end
          input_x = output_x = 0
          input_y += @stride
          output_y += 1
        end
      end
    end

    # Calls different activaton based on previous layer type

    #################################################
    # # This part is for dealing with conv layers # #

    # # Pool refers to the one dimention for window of pixels, i.e. 2 is a window of 2x2 pixels
    # def initialize(@prev_layer : ConvLayer, @pool : Int32, @stride : Int32, @logger : Logger = Logger.new(STDOUT))
    #   prev_w = prev_layer.filters.first.neurons.size # Assumes row == height
    #   new_w = ((prev_w.to_f64 - @pool.to_f64)/@stride.to_f64 + 1).to_f64
    #   raise CNNInitializationError.new("Max pool layer parameters are incorrect") unless new_w.to_i == new_w

    #   filters = prev_layer.filters.size
    #   channels = 1
    #   width = height = new_w.to_i # Assumes row == height

    #   # Channel data is stored within the filters array
    #   # This is because after convolution each filter has different feature maps
    #   @filters = Array(Array(Array(Array(Neuron)))).new(filters) {
    #     Array(Array(Array(Neuron))).new(channels) {
    #       Array(Array(Neuron)).new(height) {
    #         Array(Neuron).new(width) { Neuron.new("memory") }
    #       }
    #     }
    #   }

    #   @next_layer = DummyLayer.new
    #   @prev_layer.next_layer = self
    # end

    #######################################################################
    # # This part is for dealing with all layers other than conv layers # #

    # def _activate(prev_layer : CNNLayer)
    #   input_data = prev_layer.filters
    #   input_data.first.each_with_index do |_ch, channel|
    #     # Zoom in on a small window out of the data matrix and update
    #     input_x = input_y = output_x = output_y = 0
    #     while input_y < (input_data[0][channel].size - @pool + @stride)   # Break out of y
    #       while input_x < (input_data[0][channel].size - @pool + @stride) # Break out of x (assumes x = y)
    #         window = input_data[0][channel][input_y..(input_y + @pool - 1)].map { |row| row[input_x..(input_x + @pool - 1)].map { |neuron| neuron.activation } }

    #         @filters[0][channel][output_y][output_x].activation = window.flatten.max
    #         input_x += @stride
    #         output_x += 1
    #       end
    #       input_x = output_x = 0
    #       input_y += @stride
    #       output_y += 1
    #     end
    #   end
    # end

    # def error_prop
    #   _error_prop(@next_layer)
    # end

    # def _error_prop(next_layer : ReluLayer | DropoutLayer)
    #   @filters.size.times do |filter|
    #     @filters[filter].size.times do |channel|
    #       @filters[filter][channel].size.times do |row|
    #         @filters[filter][channel][row].size.times do |neuron|
    #           @filters[filter][channel][row][neuron].gradient = next_layer.filters[filter][channel][row][neuron].gradient
    #         end
    #       end
    #     end
    #   end
    # end

    # def _error_prop(next_layer : MaxPoolLayer)
    #   @filters.size.times do |filter|
    #     @filters[filter].size.times do |channel|
    #       input_x = input_y = output_x = output_y = 0

    #       while input_y < (@filters[filter][channel].size - @pool + @stride)   # Break out of y
    #         while input_x < (@filters[filter][channel].size - @pool + @stride) # Break out of x (assumes x = y)
    #           pool_neuron = next_layer.filters[filter][channel][output_y][output_x]

    #           # Only propagate error to the neurons that were chosen during the max pool
    #           @filters[filter][channel][input_y..(input_y + @pool - 1)].each do |row|
    #             row[input_x..(input_x + @pool - 1)].each do |neuron|
    #               if neuron.activation == pool_neuron.activation
    #                 neuron.gradient = pool_neuron.gradient
    #               end
    #             end
    #           end

    #           input_x += @stride
    #           output_x += 1
    #         end
    #         input_x = output_x = 0
    #         input_y += @stride
    #         output_y += 1
    #       end
    #     end
    #   end
    # end

    # def _error_prop(next_layer : FullyConnectedLayer)
    #   @filters.each do |filter|
    #     filter.each do |channel|
    #       channel.each do |row|
    #         row.each { |neuron| neuron.hidden_error_prop }
    #       end
    #     end
    #   end
    # end

    # def _error_prop(next_layer : InputLayer | SoftmaxLayer | DummyLayer | ConvLayer)
    #   # Do nothing
    # end

    def inspect(what : String)
      case what
      when "weights"
        puts "Maxpool layer has no weights"
      when "bias"
        puts "Maxpool layer has no bias"
      when "activations"
        @filters.each_with_index do |filter, f|
          puts "---"
          puts "Filter: #{f}, neuron activations are:"
          filter.neurons.each do |row|
            puts "#{row.map { |n| n.activation.round(4) }}"
          end
        end
      end
      puts "------------------------------------------------"
    end
  end
end
