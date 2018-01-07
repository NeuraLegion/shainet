require "logger"

module SHAInet
  alias CNN_layer = CNN_input_layer | CNV_layer # | Relu_layer | MP_layer | FC_layer | SF_layer # | DO_layer

  # # layer types:
  # input(width, height ,channels = RGB)
  # conv(width, height ,filters = features)
  # relu - same volume as previous
  # pool(width, height ,filters = features) - reduces the width and height, usually max pool
  # dropout - randomly make some neurons activaton at 0 to force new pathways
  # fc(output = classes)- single vector that clasifies, fully conneted to previous layer

  class CNN
    # getter :input_layers, :hidden_layers, :output_layers # , padding : Int32
    getter :layers

    def initialize
      @layers = Array(CNN_layer).new
      # @input_layers = Array(CNN_input_layer).new
      # @hidden_layers = Array(CNN_layer).new
      # @output_layers = Array(Layer).new
    end

    def add_input(input_volume : Array(Int32))
      @layers << CNN_input_layer.new(input_volume)
    end

    def add_conv(input_volume : Array(Int32),
                 filters_num : Int32,
                 window_size : Int32,
                 stride : Int32,
                 padding : Int32 = 0)
      @layers << CNV_layer.new(input_volume, filters_num, window_size, stride, padding)
    end

    def add_relu(prev_layer = @layers.last, l_relu_slope = 0.0)
      @layers << Relu_layer.new(prev_layer, l_relu_slope)
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
