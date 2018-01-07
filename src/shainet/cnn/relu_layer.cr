require "logger"

module SHAInet
  class Relu_layer
    getter input_data : Array(Array(Array(Float64))), l_relu_slope : Float64, output : Array(Array(Array(Float64)))

    # Add slope to initialize as leaky relu
    def initialize(input_layer : CNN_layer, @l_relu_slope : Float64 = 0.0)
      @input_data = input_layer.output

      @output = Array(Array(Array(Float64))).new(@input_data.size) {
        Array(Array(Float64)).new(@input_data.first.size) {
          Array(Float64).new(@input_data.first.first.size) { 0.0 }
        }
      }
    end

    def activate
      @input_data.size.times do |channel|
        @input_data[channel].size.times do |row|
          @input_data[channel][row].size.times do |col|
            if @l_relu_slope == 0.0
              @output[channel][row][col] = SHAInet._relu(@input_data[channel][row][col])
            else
              @output[channel][row][col] = SHAInet._l_relu(@input_data[channel][row][col], @l_relu_slope)
            end
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
      end
    end
  end
end
