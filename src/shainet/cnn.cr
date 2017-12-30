require "logger"

module SHAInet
  class CNN
    def initialize
      @layers = [] of Layer

      # # layer types:
      # input(width, height ,channels = RGB)
      # conv(width, height ,filters = features)
      # relu - same volume as previous
      # pool(width, height ,filters = features) - reduces the width and height, usually max pool
      # dropout - randomly make some neurons activaton at 0 to force new pathways
      # fc(output = classes)- single vector that clasifies, fully conneted to previous layer
    end

    def filter_layer(input : Array(Array(GenNum)), filters : Int32, window_size : Int32)
      output = [] of Array(Float64)
    end
  end

  class Conv_layer
    property :input, :output, padding : Int32, filters : Array(Array(Array(Neuron)))

    def initialize(input_volume : Array(Int32), filters_num : Int32, filter_size : Int32, stride : Int32, padding : Int32 = 0)
      unless input_volume.size == 3
        raise CNNInitializationError.new("Input volume must be an array of Int32: [width, height, channels].")
      end
      @input = Array(Array(Array(Float64))).new(input_volume.last) {
        Array(Array(Float64)).new(input_volume[1] + 2*padding) { Array(Float64).new(input_volume.first + 2*padding) { 0.0 } }
      }

      @filters = Array(Array(Array(Neuron))).new(filters_num) {
        Array(Array(Neuron)).new(filter_size) { Array(Neuron).new(filter_size) { Neuron.new(:memory) } }
      }
      @filter_size = filter_size
      @stride = stride
      @padding = padding
    end

    def convolve(input : Array(Array(Array(GenNum))))
      # Add padding to input matrices
      if @padding != 0
        input.each do |channel|
          channel.each do |row|
            @padding.times { row << 0.0 }
            @padding.times { row.insert(0, 0.0) }
          end
          padding_row = Array(Float64).new(channel.first.size) { 0.0 }
          @padding.times { channel << padding_row }
          @padding.times { channel.insert(0, padding_row) }
        end
      end

      # Use each filter to create feature map
		input.each do |channel|
		x = 0 + @padding
		y = 0 + @padding
	  	filtered_output = Array(Array(Float64)).new

			while (x <= @input.first.first.size + @padding && y <= @input.first.size + @padding) do
				array1 = channel[x..(x+@filter_size)]
				array2 = @filters[]
			end
      end
  end

  class Relu_layer
  end

  class Max_pool_layer
  end

  class Drop_out_layer
  end
end
