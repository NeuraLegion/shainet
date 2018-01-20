require "logger"

module SHAInet
  class ConvLayer
    getter prev_layer : CNNLayer | ConvLayer, filters : Array(Filter)
    getter window_size : Int32, stride : Int32, padding : Int32, activation_function : Proc(GenNum, Array(Float64))

    #################################################
    # # This part is for dealing with conv layers # #

    def initialize(@prev_layer : ConvLayer | CNNLayer,
                   filters_num : Int32 = 1,
                   @window_size : Int32 = 1,
                   @stride : Int32 = 1,
                   @padding : Int32 = 0,
                   @activation_function : Proc(GenNum, Array(Float64)) = SHAInet.none,
                   @logger : Logger = Logger.new(STDOUT))
      #
      raise CNNInitializationError.new("ConvLayer must have at least one filter") if filters_num < 1
      raise CNNInitializationError.new("Padding value must be Int32 >= 0") if @padding < 0
      raise CNNInitializationError.new("Window size value must be Int32 >= 1") if @window_size < 1
      raise CNNInitializationError.new("Stride value must be Int32 >= 1") if @stride < 1

      filters = @prev_layer.filters.size             # In conv layers channels are replaced by the feature maps,stored in the Filter class
      width = @prev_layer.filters.first.neurons.size # Assumes row == height

      # This is a calculation to make sure the input volume matches a correct desired output volume
      output_width = ((width - @window_size + 2*@padding)/@stride + 1)
      unless output_width.to_i == output_width
        raise CNNInitializationError.new("Output volume must be a whole number, change: window size, stride and/or padding")
      end

      @filters = Array(Filter).new(filters_num) { Filter.new([output_width.to_i, output_width.to_i, filters], @padding, @window_size, @stride, @activation_function) }
    end

    # Use each filter to create feature maps from the input data of the previous layer
    def activate
      @filters.each { |filter| filter.propagate_forward(@prev_layer) }
    end

    def error_prop
      @filters.each { |filter| filter.propagate_backward(@prev_layer) }
    end

    # def error_prop
    #   _error_prop(@next_layer)
    # end

    # def _error_prop(next_layer : ConvLayer)
    #   @filters.each do |filter|
    #     filter.propagate_backward(next_layer) # , activation_function)
    #   end
    # end

    # def _error_prop(next_layer : ReluLayer | DropoutLayer)
    #   @filters.size.times do |filter|
    #     @filters[filter].neurons.size.times do |row|
    #       @filters[filter].neurons[row].size.times do |neuron|
    #         @filters[filter].neurons[row][neuron].gradient = next_layer.filters[filter][0][row][neuron].gradient
    #       end
    #     end
    #   end
    # end

    # def _error_prop(next_layer : MaxPoolLayer)
    #   @filters.size.times do |filter|
    #     input_x = input_y = output_x = output_y = 0

    #     while input_y < (@filters[filter].neurons.size - next_layer.pool + next_layer.stride)   # Break out of y
    #       while input_x < (@filters[filter].neurons.size - next_layer.pool + next_layer.stride) # Break out of x (assumes x = y)
    #         pool_neuron = next_layer.filters[filter][0][output_y][output_x]

    #         # Only propagate error to the neurons that were chosen during the max pool
    #         @filters[filter].neurons[input_y..(input_y + next_layer.pool - 1)].each do |row|
    #           row[input_x..(input_x + next_layer.pool - 1)].each do |neuron|
    #             if neuron.activation == pool_neuron.activation
    #               neuron.gradient = pool_neuron.gradient
    #             end
    #           end

    #           input_x += next_layer.stride
    #           output_x += 1
    #         end
    #         input_x = output_x = 0
    #         input_y += next_layer.stride
    #         output_y += 1
    #       end
    #     end
    #   end
    # end

    # def _error_prop(next_layer : FullyConnectedLayer)
    #   @filters.each do |filter|
    #     filter.neurons.each do |row|
    #       row.each { |neuron| neuron.hidden_error_prop }
    #     end
    #   end
    # end

    # def _error_prop(next_layer : InputLayer | SoftmaxLayer | DummyLayer)
    #   # Do nothing
    # end

    def inspect(what : String)
      case what
      when "weights"
        @filters.each_with_index do |filter, i|
          puts "---"
          puts "Filter #{i}, weights:"
          filter.synapses.each_with_index do |channel, j|
            puts "Channel: #{j}"
            channel.each { |row| puts "#{row.map { |syn| syn.weight.round(4) }}" }
          end
        end
      when "bias"
        @filters.each_with_index { |filter, i| puts "Filter #{i}, bias:#{filter.bias.round(4)}" }
      when "activations"
        @filters.each_with_index do |filter, f|
          puts "---"
          puts "Filter: #{f}, neuron activations are:"
          filter.neurons.each do |row|
            puts "#{row.map { |n| n.activation.round(4) }}"
          end
        end
      when "gradients"
        @filters.each_with_index do |filter, f|
          puts "---"
          puts "Filter: #{f}, neuron gradients are:"
          filter.neurons.each do |row|
            puts "#{row.map { |n| n.gradient.round(4) }}"
          end
        end
      end
      puts "------------------------------------------------"
    end
  end
end
