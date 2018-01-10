require "logger"
require "./**"

module SHAInet
  alias CNNLayer = InputLayer | ReluLayer | MaxPoolLayer | FullyConnectedLayer | DropoutLayer

  class CNN
    getter :layers

    def initialize
      @layers = Array(CNNLayer | ConvLayer).new
    end

    def add_input(input_volume : Array(Int32))
      @layers << InputLayer.new(input_volume)
    end

    def add_conv(filters_num : Int32 = 1,
                 window_size : Int32 = 1,
                 stride : Int32 = 1,
                 padding : Int32 = 0)
      @layers << ConvLayer.new(@layers.last, filters_num, window_size, stride, padding)
    end

    def add_relu(l_relu_slope = 0.0)
      @layers << ReluLayer.new(@layers.last, l_relu_slope)
    end

    def add_maxpool(pool : Int32, stride : Int32)
      @layers << MaxPoolLayer.new(@layers.last, pool, stride)
    end

    def add_fconnect(l_size : Int32, softmax = false, activation_function : Proc(GenNum, Array(Float64)) = SHAInet.sigmoid)
      @layers << FullyConnectedLayer.new(@layers.last, l_size, softmax = false, activation_function)
    end

    def add_dropout(drop_percent : Int32 = 5)
      @layers << DropoutLayer.new(@layers.last, drop_percent)
    end

    def run(input_data : Array(Array(Array(GenNum))), stealth : Bool = false)
      if stealth == false
        puts "############################"
        puts "Starting run..."
      end
      # Activate all layers one by one
      @layers.each_with_index do |layer, i|
        if layer.is_a?(InputLayer)
          layer.as(InputLayer).activate(input_data) # activation of input layer
        else
          layer.activate # activate the rest of the layers
        end
      end
      # Get the result from the output layer
      puts "Network output:"
      puts @layers.last.as(FullyConnectedLayer).output
      puts "Finished run."
      puts "############################"
    end
  end
end
