require "logger"

module SHAInet
  class CNN_input_layer
    property :neurons
    getter :output

    def initialize(input_volume : Array(Int32))
      unless input_volume.size == 3
        raise CNNInitializationError.new("Input volume must be an array of Int32: [width, height, channels].")
      end

      unless input_volume[0] == input_volume[1]
        raise CNNInitializationError.new("Width and height of input must be of the same size.")
      end

      @neurons = Array(Array(Array(Neuron))).new(input_volume[2]) {
        Array(Array(Neuron)).new(input_volume[0]) {
          Array(Neuron).new(input_volume[1]) { Neuron.new("memory") }
        }
      }

      @output = Array(Array(Array(Float64))).new
    end

    def activate(input_data : Array(Array(Array(GenNum))))
      # Input the data into the first layer
      @output = input_data.as(Array(Array(Array(Float64)))).dup
      puts "Input data: #{input_data}"
      input_data.size.times do |channel|
        input_data[channel].size.times do |row|
          input_data[channel][row].size.times do |col|
            neurons[channel][row][col].activation = input_data[channel][row][col]
            # TODO: Add multiple input layer support
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
        @neurons.each_with_index do |channel, ch|
          puts "Channel: #{ch}"
          channel.each do |row|
            puts "#{row.map { |n| n.activation }}"
          end
        end
      end
      puts "------------"
    end
  end
end
