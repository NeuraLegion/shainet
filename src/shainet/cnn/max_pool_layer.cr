require "logger"

module SHAInet
  class MP_layer
    getter input_data : Array(Array(Array(Float64))), output : Array(Array(Array(Float64))), pool : Int32

    # Pool refers to the whole window of pixels, 4 is a window of 2x2 pixels
    def initialize(input_layer : CNN_layer, @pool : Int32)
      @input_data = input_layer.output

      @output = Array(Array(Array(Float64))).new(@input_data.size) {
        Array(Array(Float64)).new(@input_data.first.size/(@pool*0.5)) {
          Array(Float64).new(@input_data.first.first.size/(@pool*0.5)) { 0.0 }
        }
      }
    end

    def activate
      @input_data.size.times do |channel|
        input_x = input_y = output_x = output_y = 0

        # Zoom in on a small window out of the data matrix and update
        until input_y == (@input_data[channel].size - @pool*0.5 - 1)
          until input_x == (@input_data[channel][input_y].size - @pool*0.5 - 1)
            window = @input_data[channel][input_y..(input_y + @pool*0.5 - 1)].map { |m| m[input_x..(input_x + @pool*0.5 - 1)] }
            @output[channel][output_y][output_x] = window.flatten.max
            input_x += @pool*0.5
            output_x += 1
          end
          input_x = output_x = 0
          input_y += @pool*0.5
          output_y += 1
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
