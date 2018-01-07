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
    cnn.add_input(volume = [7, 7, 2])
    # Conv layer needs: input_volume, filters_num, window_size, stride, padding
    cnn.add_conv(volume = [7, 7, 2], filters = 1, window_size = 3, stride = 1, padding = 1)

    # cnn.layers.each { |l| p l.class }

    # cnn.layers.last.output.each_with_index { |channel, i| puts "#Channel #{i}: #{channel}" }
    # #_with_index do |channel, j|
    #   channel.each_with_index do |x, i|
    #     if i % channel.size == 0
    #       puts "Channel #{j}:"
    #       puts x
    #     else
    #       puts x
    #     end
    #   end
    # end

    # prints filter weights
    # cnn.layers.each { |layer| layer.inspect("weights") }

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
