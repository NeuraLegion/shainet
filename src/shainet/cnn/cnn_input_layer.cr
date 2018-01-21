require "logger"

module SHAInet
  class InputLayer
    getter filters : Array(Filter)

    def initialize(input_volume : Array(Int32), @logger : Logger = Logger.new(STDOUT))
      unless input_volume.size == 3
        raise CNNInitializationError.new("Input volume must be an array of Int32: [width, height, channels].")
      end

      unless input_volume[0] == input_volume[1]
        raise CNNInitializationError.new("Width and height of input must be of the same size.")
      end

      # Channel data is stored within the Filter class, this is needed for smooth work with all other layers.
      channels = input_volume[2] # filters == channels
      width = input_volume[0]
      height = input_volume[1]

      @filters = Array(Filter).new(channels) { Filter.new([width, height, 1]) }
    end

    def activate(input_data : Array(Array(Array(GenNum))))
      # Input the data into the first layer
      input_data.size.times do |channel|
        input_data[channel].size.times do |row|
          input_data[channel][row].size.times do |col|
            @filters[channel].neurons[row][col].activation = input_data[channel][row][col].to_f64
          end
        end
      end
    end

    def error_prop
      # Do nothing
    end

    def update_wb(learn_type : Symbol | String, batch : Bool = false)
      # Do nothing
    end

    def inspect(what : String)
      case what
      when "weights"
        puts "input layer has no weights"
      when "bias"
        puts "input layer has no bias"
      when "activations"
        @filters.each_with_index do |filter, f|
          puts "---"
          puts "Channel: #{f}, neuron activations are:"
          filter.neurons.each do |row|
            puts "#{row.map { |n| n.activation.round(4) }}"
          end
        end
      when "gradients"
        puts "input layer has no gradients"
      end
      puts "------------------------------------------------"
    end
  end
end
