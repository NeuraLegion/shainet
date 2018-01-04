require "./spec_helper"
require "csv"

# Extract train data
system("cd #{__DIR__}/test_data && tar xvf tests.tar.gz")

describe SHAInet::CNN do
  it "Check basic cnn features" do
    data = [
      [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],

      [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
    ]
    cnn = SHAInet::CNN.new
    # Input layer needs: volume = [width, height, channels]
    cnn.add_l_input(volume = [7, 7, 2])
    # Conv layer needs: input_volume, filters_num, window_size, stride, padding
    cnn.add_l_conv(volume = [7, 7, 2], filters = 1, window_size = 3, stride = 1, padding = 1)

    cnn.hidden_layers.first.output.each_with_index do |channel, j|
      channel.each_with_index do |x, i|
        if i % channel.size == 0
          puts "Channel #{j}:"
          puts x
        else
          puts x
        end
      end
    end

    # prints filter weights
    puts "Weights:"
    cnn.hidden_layers.first.filters.first.receptive_field.weights.each { |w| puts w }

    cnn.run(data)

    # cnn.hidden_layers.first.output.each_with_index do |channel, j|
    #   channel.each_with_index do |x, i|
    #     if i % channel.size == 0
    #       puts "Channel #{j}:"
    #       puts x
    #     else
    #       puts x
    #     end
    #   end
    # end
    #
  end
  #
end
