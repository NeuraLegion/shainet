require "logger"

module SHAInet
  class CNV_layer
    getter filters : Array(Filter), window_size : Int32, stride : Int32, padding : Int32, output : Array(Array(Array(Float64)))

    def initialize(input_volume : Array(Int32),
                   filters_num : Int32,
                   @window_size : Int32,
                   @stride : Int32,
                   @padding : Int32 = 0)
      raise CNNInitializationError.new("Padding value must be Int32 >= 0") if @padding < 0

      unless input_volume.size == 3
        raise CNNInitializationError.new("Input volume must be an array of Int32: [width, height, channels].")
      end

      output_surface = ((input_volume[0] - @window_size + 2*@padding)/@stride + 1)

      unless output_surface.class == Int32
        raise CNNInitializationError.new("Output volume must be a whole number, change: window size or stride or padding")
      end

      @filters = Array(Filter).new(filters_num) { Filter.new([input_volume[0], input_volume[1]], window_size) }

      @output = Array(Array(Array(Float64))).new(filters_num) {
        Array(Array(Float64)).new(output_surface) {
          Array(Float64).new(output_surface) { 0.0 }
        }
      }
    end

    def pad(input_data : Array(Array(Array(Neuron))))
      if @padding == 0
        return input_data
      else
        blank_neuron = Neuron.new("memory")
        padded_data = input_data
        padded_data.each do |channel|
          channel.each do |row|
            @padding.times { row << blank_neuron }
            @padding.times { row.insert(0, blank_neuron) }
          end
          padding_row = Array(Neuron).new(channel.first.size) { blank_neuron }
          @padding.times { channel << padding_row }
          @padding.times { channel.insert(0, padding_row) }
        end
        return padded_data
      end
    end

    # Use each filter to create feature map
    def activate(input_layer : CNN_input_layer | CNN_layer)
      padded_data = pad(input_layer.as(CNN_input_layer).neurons)

      padded_data.size.times do |channel|
        @filters.each do |filter|
          input_x = input_y = output_x = output_y = 0

          # Zoom in on a small window out of the data matrix and update
          loop do
            # # TODO ##
            # # Fix breaking logic in loop
            loop do
              break if output_x == @output.first.first.size
              window = padded_data[channel][input_y..(input_y + @window_size - 1)].map { |m| m[input_x..(input_x + @window_size - 1)] }
              @output[channel][output_y][output_x] = filter.receptive_field.prpogate_forward(window, filter.neurons[output_y][output_x]) rescue break
              puts "row: #{input_y} col: #{input_x} window: #{window.each { |r| puts r.map { |n| n.activation } }}"
              input_x += @stride
              output_x += 1
            end
            input_x = output_x = 0
            input_y += @stride
            output_y += 1
            break if output_y == @output.first.size
          end

          puts "-------------"
        end
      end
    rescue e : Exception
      puts "#{e.message}, #{e.inspect_with_backtrace}"
    end

    def inspect(what : String)
      case what
      when "weights"
        filters.each_with_index do |filter, i|
          puts "Filter #{i}, weights:"
          p filter.receptive_field.weights
        end
      when "bias"
        filters.each_with_index do |filter, i|
          puts "Filter #{i}, bias:"
          p filter.receptive_field.bias
        end
      when "neurons"
        filters.each_with_index do |filter, i|
          puts "Filter #{i}, neurons:"
          p filter.neurons
        end
      end
    end
  end

  # In CNNs a filter is a separate unit within the layer
  class Filter
    getter neurons : Array(Array(Neuron)), receptive_field : Receptive_field

    def initialize(input_surface : Array(Int32), window_size : Int32)
      @neurons = Array(Array(Neuron)).new(input_surface[1]) { Array(Neuron).new(input_surface[0]) { Neuron.new("memory") } }
      @receptive_field = Receptive_field.new(window_size)
    end
  end

  # This is somewhat similar to a synapse
  class Receptive_field
    property weights : Array(Array(Float64)), bias : Float64
    getter :window_size

    def initialize(@window_size : Int32)
      @weights = Array(Array(Float64)).new(@window_size) { Array(Float64).new(@window_size) { rand(0.0..1.0).to_f64 } }
      @bias = rand(-1..1).to_f64
    end

    # Takes a small window from the input data to preform feed forward
    def prpogate_forward(input_window : Array(Array(Neuron)), target_neuron : Neuron)
      weighted_sum = Float64.new(0)
      @weights.size.times do |row|
        @weights.size.times do |col|
          weighted_sum += input_window[row][col].activation*@weights[row][col]
        end
      end
      target_neuron.activation = weighted_sum + @bias
      return target_neuron.activation
    end

    def prpogate_backward
    end
  end
end
