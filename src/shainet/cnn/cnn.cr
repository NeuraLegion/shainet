require "logger"

module SHAInet
  alias CNN_layer = Conv_layer | Relu_layer | Max_pool_layer | FC_layer

  # # layer types:
  # input(width, height ,channels = RGB)
  # conv(width, height ,filters = features)
  # relu - same volume as previous
  # pool(width, height ,filters = features) - reduces the width and height, usually max pool
  # dropout - randomly make some neurons activaton at 0 to force new pathways
  # fc(output = classes)- single vector that clasifies, fully conneted to previous layer

  class CNN
    getter :input_layers, :hidden_layers, :output_layers # , padding : Int32

    def initialize
      @input_layers = Array(CNN_input_layer).new
      @hidden_layers = Array(CNN_layer).new
      @output_layers = Array(Layer).new
    end

    def add_l_input(input_volume : Array(Int32))
      @input_layers << CNN_input_layer.new(input_volume)
    end

    def add_l_conv(input_volume : Array(Int32),
                   filters_num : Int32,
                   window_size : Int32,
                   stride : Int32,
                   padding : Int32 = 0)
      @hidden_layers << Conv_layer.new(input_volume, filters_num, window_size, stride, padding)
    end

    def run(input_data : Array(Array(Array(GenNum))))
      # Input the data into the first layer
      input_data.size.times do |channel|
        channel.times do |row|
          row.times do |col|
            @input_layers.first.neurons[channel][row][col].activation = input_data[channel][row][col]
          end
        end
      end

      # Activate all hidden layers one by one
      @hidden_layers.size.times do |l|
        if l == 0
          @hidden_layers[l].activate(@input_layers.last)
          # else
          #   @hidden_layers[l].activate(@hidden_layers[l - 1])
        end
      end
    end
  end

  class Max_pool_layer
  end

  # class Drop_out_layer
  # end

  # class FC_layer
  # end
end
