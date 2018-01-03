require "logger"

module SHAInet
  alias CNN_layer = Conv_layer | Relu_layer | Max_pool_layer | FC_layer

  class CNN < Network
    def initialize
      @input_layers = Array(CNN_input_layer).new
      @hidden_layers = Array(CNN_layer).new
      @output_layers = Array(Layer).new

      # # layer types:
      # input(width, height ,channels = RGB)
      # conv(width, height ,filters = features)
      # relu - same volume as previous
      # pool(width, height ,filters = features) - reduces the width and height, usually max pool
      # dropout - randomly make some neurons activaton at 0 to force new pathways
      # fc(output = classes)- single vector that clasifies, fully conneted to previous layer
    end

    def pad(input : Array(Array(Array(GenNum))), padding : Int32)
      if padding < 1
        raise CNNInitializationError.new("Padding value must be Int32 > 0")
      end
      input.each do |channel|
        channel.each do |row|
          padding.times { row << 0.0 }
          padding.times { row.insert(0, 0.0) }
        end
        padding_row = Array(Float64).new(channel.first.size) { 0.0 }
        padding.times { channel << padding_row }
        padding.times { channel.insert(0, padding_row) }
      end
      return input
    end

    def receptive_field_split(input : Array(Array(Array(GenNum))), filter_size : Int32, stride : Int32)
      vision_windows = Array(Layer).new

      x = y = 0
    end
  end

  class CNN_neuron < Neuron
  end

  class CNN_synapse < Synapse
  end

  class CNN_input_layer
    def initialize(input_volume : Array(Int32))
      unless input_volume.size == 3
        raise CNNInitializationError.new("Input volume must be an array of Int32: [width, height, channels].")
      end

      unless input_volume[0].size == input_volume[1].size
        raise CNNInitializationError.new("Width and height of input must be of the same size.")
      end

      @channels = Array(Array(Array(Neuron))).new(input_volume[2]) {
        Array(Array(Neuron)).new(input_volume[0]) { Array(Neuron).new(input_volume[1]) { Neuron.new(:memory) } }
      }
    end
  end

  class Conv_layer
  end

  class Receptive_field
    property window_loacation : Array(Float64), synapses : Array(Array(CNN_synapse)), bias : Float64
    property :source_neurons, :target_neuron
    getter :window_size

    def initialize(@window_size : Int32)
      @window_loacation = [0, 0]
      @synapses = Array(Array(CNN_synapse)).new(@window_size) { Array(CNN_synapse).new(@window_size) { CNN_synapse.new } }
      @bias = rand(-1..1).to_f64

      @source_neurons = nil
      @target_neuron = nil
    end

    def prpogate_forward(input_matrix : Array(Array(Neuron)), window_loacation : Array(Float64), target_neuron : CNN_neuron)
      weighted_sum = Float64.new(0)
      @synapses.size.each do |row|
        row.size.each do |col|
          weighted_sum += input_matrix[window_loacation[0] + row].activaton*@synapses[row][col].weight
        end
      end
      target_neuron.activation = weighted_sum + @bias
    end

    def prpogate_backward
    end
  end

  class Relu_layer
  end

  class Max_pool_layer
  end

  class Drop_out_layer
  end

  class FC_layer
  end
end

def initialize(input_volume : Array(Int32), filters_num : Int32, filter_size : Int32, stride : Int32, padding : Int32 = 0)
  unless input_volume.size == 3
    raise CNNInitializationError.new("Input volume must be an array of Int32: [width, height, channels].")
  end

  @filters = Array(Array(Array(Neuron))).new(filters_num) {
    Array(Array(Neuron)).new(filter_size) { Array(Neuron).new(filter_size) { Neuron.new(:memory) } }
  }

  @output = Array(Array(Array(Float64))).new(filters_num) {
    Array(Array(Float64)).new(input_volume[0]) { Array(Float64).new(input_volume[1]) { 0.0 } }
  }

  @filter_size = filter_size
  @stride = stride
  @padding = padding
end

def convolve(input : Array(Array(Array(GenNum))))
  # Use each filter to create feature map

  # @filters.each do |matrix|
  #  row = 0 + @padding
  #  col = 0 + @padding

  #  # while

  #  	while (row <= input.first.first.size + @padding && y <= input.first.size + @padding)
  # 	(row..row + @stride).each do |x|
  # 		(col..col + @stride).each do |y|

  #   	filtered_output_matrices = Array(Array(Array(Array(GenNum)))).new
  #   	@filters.each do |filter|
  #   		output_matrix = Array(Array(GenNum))
  #   		filtered_rows = Array(Float64).new
  #   		filter.each_with_index do |row|
  # 	  		array1 = channel[y+i][x..(x + @filter_size -1)]
  # 	  		array2 = matrix[i]
  # 	  		new_array = vector_mult(array1,array2) # + bias
  # 	  		array_sum = new_array.reduce {|acc,i| acc + i}
  # 	  		filtered_rows << array_sum
  # 	  	end

  # 	end
  # end
end
