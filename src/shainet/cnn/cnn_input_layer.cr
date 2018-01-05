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

      @output = Array(Array(Array(GenNum))).new(input_volume[2]) {
        Array(Array(GenNum)).new(input_volume[0]) { Array(GenNum).new(input_volume[1]) { 0.0 } }
      }
    end

    def activate(target : CNN_layer)
      # test
    end
  end
end
