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

    # Input layer params: volume = [width, height, channels]
    cnn.add_input(volume = [7, 7, 2])
    cnn.layers.last.inspect("activations")
    # Conv layer params: filters_num, window_size (one dimentional, i.e 2 = 2x2 window), stride, padding
    cnn.add_conv(filters = 1, window_size = 3, stride = 1, padding = 1)
    cnn.layers.last.inspect("activations")
    cnn.layers.last.inspect("weights")
    # Relu layer params: slope (default is set to 0.0, change for leaky relu)
    cnn.add_relu
    cnn.layers.last.inspect("activations")
    # Pool layer params: pool size (one dimentional, i.e 2 = 2x2 pool), stride
    cnn.add_maxpool(pool = 2, stride = 1)
    cnn.layers.last.inspect("activations")
    # Fully conncet layer params: l_size, activation_function (default is SHAInet.sigmoid, use SHAInet.softmax when a softmax layer is needed)
    cnn.add_fconnect(10)
    cnn.add_fconnect(5, SHAInet.softmax)
    # cnn.layers.each { |l| p l.class }

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
