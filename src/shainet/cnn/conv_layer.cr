require "logger"

module SHAInet
  # In conv layers a filter is a separate unit within the layer, with special parameters
  class Filter
    getter input_surface : Array(Int32), window_size : Int32
    property neurons : Array(Array(Neuron)), receptive_field : ReceptiveField

    def initialize(@input_surface : Array(Int32), # expecting [width, height, channels]
                   @window_size : Int32)
      #
      @neurons = Array(Array(Neuron)).new(input_surface[1]) {
        Array(Neuron).new(input_surface[0]) { Neuron.new("memory") }
      }
      @receptive_field = ReceptiveField.new(window_size, input_surface[2])
    end

    def clone
      filter_old = self
      filter_new = Filter.new(filter_old.input_surface, filter_old.window_size)

      filter_new.neurons = filter_old.neurons.clone
      filter_new.receptive_field = filter_old.receptive_field.clone
      return filter_new
    end
  end

  # This is somewhat similar to a synapse
  class ReceptiveField
    property weights : Array(Array(Array(Float64))), bias : Float64
    getter window_size : Int32, channels : Int32

    def initialize(@window_size : Int32, @channels : Int32)
      @weights = Array(Array(Array(Float64))).new(channels) {
        Array(Array(Float64)).new(@window_size) {
          Array(Float64).new(@window_size) { rand(0.0..1.0).to_f64 }
        }
      }
      @bias = rand(-1..1).to_f64
    end

    def clone
      rf_old = self
      rf_new = ReceptiveField.new(rf_old.window_size, rf_old.channels)
      rf_new.weights = rf_old.weights
      rf_new.bias = rf_old.bias

      return rf_new
    end

    # Takes a small window from the input data (CxHxW) to preform feed forward
    # Propagate forward from CNNLayer
    def prpogate_forward(input_window : Array(Array(Array(Neuron))), target_neuron : Neuron)
      weighted_sum = Float64.new(0)
      @weights.each_with_index do |_c, channel|
        @weights[channel].each_with_index do |_r, row|
          @weights[channel][row].each_with_index do |_c, col|
            weighted_sum += input_window[channel][row][col].activation * @weights[channel][row][col]
          end
        end
      end
      target_neuron.activation = weighted_sum + @bias
    end

    # Propagate forward from ConvLayer
    def prpogate_forward(input_window : Array(Array(Neuron)), target_neuron : Neuron)
      weighted_sum = Float64.new(0)
      @weights.first.size.times do |row|
        @weights.first[row].size.times do |col|
          weighted_sum += input_window[row][col].activation*@weights.first[row][col]
        end
      end
      target_neuron.activation = weighted_sum + @bias
    end

    def prpogate_backward
    end
  end

  class ConvLayer
    getter input_layer : CNNLayer | ConvLayer, filters : Array(Filter), window_size : Int32, stride : Int32, padding : Int32

    #################################################
    # # This part is for dealing with conv layers # #

    def initialize(input_layer : ConvLayer,
                   filters_num : Int32,
                   @window_size : Int32,
                   @stride : Int32,
                   @padding : Int32 = 0,
                   @logger : Logger = Logger.new(STDOUT))
      #
      raise CNNInitializationError.new("ConvLayer must have at least one filter") if filters_num < 1
      raise CNNInitializationError.new("Padding value must be Int32 >= 0") if @padding < 0
      raise CNNInitializationError.new("Window size value must be Int32 >= 1") if @window_size < 1
      raise CNNInitializationError.new("Stride value must be Int32 >= 1") if @stride < 1

      channels = 1                                            # In conv layers channels is always 1, but have may multiple filters
      width = height = input_layer.filters.first.neurons.size # Assumes row == height
      @input_layer = input_layer

      # This is a calculation to make sure the input volume matches a correct desired output volume
      output_surface = ((width - @window_size + 2*@padding)/@stride + 1)
      unless output_surface.class == Int32
        raise CNNInitializationError.new("Output volume must be a whole number, change: window size or stride or padding")
      end

      @filters = Array(Filter).new(filters_num) { Filter.new([width, height, channels], @window_size) }
    end

    # Adds padding to all Filters of input data
    def _pad(input_layer : ConvLayer)
      input_data = input_layer.filters.clone # Array of filter class

      if @padding == 0
        return input_data
      else
        blank_neuron = Neuron.new("memory")
        blank_neuron.activation = 0.0
        padded_data = input_data.dup

        padded_data.each do |filter|
          # Add padding at the sides
          filter.neurons.each do |row|
            @padding.times { row << blank_neuron }
            @padding.times { row.insert(0, blank_neuron) }
          end
          # Add padding at the top/bottom
          padding_row = Array(Neuron).new(filter.neurons.first.size) { blank_neuron }
          @padding.times { filter.neurons << padding_row }
          @padding.times { filter.neurons.insert(0, padding_row) }
        end
        return padded_data
      end
    end

    def _convolve(input_layer : ConvLayer)
      padded_data = _pad(input_layer) # Array of filter class

      @filters.each_with_index do |_f, self_filter|
        puts "Filter: #{self_filter}"
        # Zoom in on a small window out of the input data volume and update each neuron of the filter
        padded_data.size.times do |filter|
          input_x = input_y = output_x = output_y = 0

          while input_y < (padded_data.first.neurons.size - @window_size + @stride)         # Break out of y
            while input_x < (padded_data.first.neurons.first.size - @window_size + @stride) # Break out of x (assumes x = y)
              window = padded_data[filter].neurons[input_y..(input_y + @window_size - 1)].map { |n| n[input_x..(input_x + @window_size - 1)] }
              target_neuron = @filters[self_filter].neurons[output_y][output_x]
              @filters[self_filter].receptive_field.prpogate_forward(window, target_neuron)
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

    # Use each filter to create feature maps for the input data
    def _activate(input_layer : ConvLayer)
      _convolve(@input_layer)
    end

    #######################################################################
    # # This part is for dealing with all layers other than conv layers # #

    def initialize(input_layer : CNNLayer,
                   filters_num : Int32,
                   @window_size : Int32,
                   @stride : Int32,
                   @padding : Int32 = 0,
                   @logger : Logger = Logger.new(STDOUT))
      #
      raise CNNInitializationError.new("ConvLayer must have at least one filter") if filters_num < 1
      raise CNNInitializationError.new("Padding value must be Int32 >= 0") if @padding < 0
      raise CNNInitializationError.new("Window size value must be Int32 >= 1") if @window_size < 1
      raise CNNInitializationError.new("Stride value must be Int32 >= 1") if @stride < 1

      # In other layers filters is always 1, but may have multiple channels
      channels = input_layer.filters.first.size
      width = height = input_layer.filters.first.first.size # Assumes row == height
      @input_layer = input_layer

      # This is a calculation to make sure the input volume matches a correct desired output volume
      output_surface = ((width.to_f64 - @window_size.to_f64 + 2*@padding.to_f64)/@stride.to_f64 + 1).to_f64
      unless output_surface.to_i == output_surface
        raise CNNInitializationError.new("Output volume must be a whole number, change: window size or stride or padding")
      end

      @filters = Array(Filter).new(filters_num) { Filter.new([width, height, channels], @window_size) }
    end

    # Adds padding to all channels of input data
    def _pad(input_layer : CNNLayer, print : Bool)
      input_data = input_layer.filters.first.clone # Array of all channels
      if @padding == 0
        return input_data
      else
        blank_neuron = Neuron.new("memory")
        blank_neuron.activation = 0.0
        padded_data = input_data

        # Go over each channel and add padding
        padded_data.size.times do |channel|
          # Add padding at the sides
          padded_data[channel].each do |row|
            @padding.times { row << blank_neuron }
            @padding.times { row.insert(0, blank_neuron) }
          end
          # Add padding at the top/bottom
          padding_row = Array(Neuron).new(padded_data.first.first.size) { blank_neuron }
          @padding.times { padded_data[channel] << padding_row }
          @padding.times { padded_data[channel].insert(0, padding_row) }
        end
        if print == true
          padded_data.each_with_index do |channel, ch|
            puts "padded_data:"
            puts "Channel: #{ch}"
            channel.each do |row|
              puts "#{row.map { |n| n.activation }}"
            end
          end
        end
        return padded_data
      end
    end

    def _convolve(input_layer : CNNLayer)
      padded_data = _pad(input_layer, print = false) # Array of all channels

      @filters.each_with_index do |_f, self_filter|
        # Zoom in on a small window out of the input data volume and update each neuron of the filter
        input_x = input_y = output_x = output_y = 0

        while input_y < (padded_data.first.size - @window_size + @stride)         # Break out of y
          while input_x < (padded_data.first.first.size - @window_size + @stride) # Break out of x (assumes x = y)
            window = padded_data.map { |channel| channel[input_y..(input_y + @window_size - 1)].map { |n| n[input_x..(input_x + @window_size - 1)] } }
            target_neuron = @filters[self_filter].neurons[output_y][output_x]
            @filters[self_filter].receptive_field.prpogate_forward(window, target_neuron)
            input_x += @stride
            output_x += 1
          end
          input_x = output_x = 0
          input_y += @stride
          output_y += 1
        end
      end
    end

    # Use each filter to create feature maps for the input data
    def _activate(input_layer : CNNLayer)
      _convolve(@input_layer)
    end

    #########################
    # # General functions # #

    # Calls different activaton based on previous layer type
    def activate
      _activate(@input_layer)
    end

    def inspect(what : String)
      puts "Conv layer:"
      case what
      when "weights"
        filters.each_with_index do |filter, i|
          puts "Filter #{i}, weights:"
          filter.receptive_field.weights.each_with_index do |channel, j|
            puts "Channel: #{j}"
            channel.each { |row| puts "#{row.map { |w| }}" }
          end
        end
      when "bias"
        filters.each_with_index { |filter, i| puts "Filter #{i}, bias:#{filter.receptive_field.bias}" }
      when "activations"
        @filters.each_with_index do |filter, i|
          puts "Filter #{i}, activations:"
          filter.neurons.each { |row| puts "#{row.map { |n| n.activation }}" }
        end
      end
      puts "------------"
    end
  end
end
