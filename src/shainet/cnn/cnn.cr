require "logger"
require "./**"

module SHAInet
  alias CNNLayer = InputLayer | ReluLayer | MaxPoolLayer | FullyConnectedLayer | DropoutLayer
  alias CNNLayerClass = InputLayer.class | ReluLayer.class | MaxPoolLayer.class | FullyConnectedLayer.class | DropoutLayer.class

  # # layer types:
  # input(width, height ,channels = RGB)
  # conv(width, height ,filters = features)
  # relu - same volume as previous
  # pool(width, height ,filters = features) - reduces the width and height, usually max pool
  # dropout - randomly make some neurons activaton at 0 to force new pathways
  # fc(output = classes)- single vector that clasifies, fully conneted to previous layer

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

    def add_fconnect(l_size : Int32, activation_function : Proc(GenNum, Array(Float64)) = SHAInet.sigmoid)
      @layers << FullyConnectedLayer.new(@layers.last, l_size, activation_function)
    end

    def add_dropout(drop_percent : Int32 = 5)
      @layers << DropoutLayer.new(@layers.last, drop_percent)
    end

    def run(input_data : Array(Array(Array(GenNum))))
      # Activate all hidden layers one by one
      @layers.each_with_index do |l, i|
        if l.is_a?(CNN_input_layer)
          l.as(CNN_input_layer).activate(input_data)
        else
          unless l.is_a?(CNN_input_layer)
            l.activate(@layers[i - 1])
          end
        end
      end

      # Get the result from the output layer
      @layers.last.output
    end
  end
end
