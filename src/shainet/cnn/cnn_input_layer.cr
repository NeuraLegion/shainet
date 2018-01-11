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
            @filters.first[channel][row][col].activation = input_data[channel][row][col]
            # TODO: Add multiple input layer support
          end
        end
      end
    end

    def inspect(what : String)
      puts "Input layer:"
      case what
      when "weights"
        puts "input layer has no wights"
      when "bias"
        puts "input layer has no bias"
      when "activations"
        @filters.first.each_with_index do |channel, ch|
          puts "Channel: #{ch}, neuron activations are:"
          channel.each do |row|
            puts "#{row.map { |n| n.activation }}"
          end
        end
      end
      puts "------------"
    end
  end
end
