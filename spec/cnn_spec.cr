require "./spec_helper"
require "csv"

# Extract train data
system("cd #{__DIR__}/test_data && tar xvf tests.tar.gz")

describe SHAInet::CNN do
  it "Check basic cnn features" do
    img_data = [
      [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],

      [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
       [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
       [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
       [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
       [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
       [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
       [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]],
    ]
    expected_output = [0.0, 0.0, 0.0, 0.0, 1.0]

    training_data = [[img_data, expected_output]]

    cnn = SHAInet::CNN.new

    cnn.add_input(volume = [7, 7, 2])
    cnn.add_conv(filters = 2, window_size = 3, stride = 1, padding = 1, activation_function = SHAInet.none)
    cnn.add_conv(filters = 2, window_size = 3, stride = 1, padding = 1, activation_function = SHAInet.sigmoid)
    cnn.add_relu
    cnn.add_maxpool(pool = 2, stride = 1)
    cnn.add_fconnect(l_size = 10, SHAInet.sigmoid)
    cnn.add_dropout(drop_percent = 10)
    cnn.add_fconnect(l_size = 5, SHAInet.none)
    cnn.add_softmax
    # Input layer params: volume = [width, height, channels]
    # Conv layer params: filters_num, window_size (one dimentional, i.e 2 = 2x2 window), stride, padding
    # Relu layer params: slope (default is set to 0.0, change for leaky relu)
    # Pool layer params: pool size (one dimentional, i.e 2 = 2x2 pool), stride
    # Fully conncet layer params: l_size, activation_function (default is SHAInet.sigmoid, use SHAInet.softmax when a softmax layer is needed)
    # Softmax layer: can only come after a FC layer, has the same output size as previous layer

    # Run the network with a single input
    cnn.run(img_data, stealth = true)

    # cnn.evaluate(img_data, expected_output, :mse)

    # cnn.layers.each_with_index do |layer, i|
    #   puts "Layer #{i} - #{layer.class}:"
    #   layer.inspect("activations")
    # end

    # cnn.layers.each_with_index { |layer, i| puts "Layer #{i}:", layer.inspect("weights") }
    # puts "-----"
    puts "Network output:\n#{cnn.layers.last.as(SHAInet::FullyConnectedLayer | SHAInet::SoftmaxLayer).output}\n"
    puts "Error signal is:\n#{cnn.error_signal}"

    # puts "-----"
    # cnn.train(training_data, training_type = :rprop, cost = :mse, epochs = 20000, threshold = 0.000001, log_each = 1000)

    cnn.train_batch(training_data, training_type = :rprop, cost = :mse, epochs = 20000, threshold = 0.000001, log_each = 1000, minib = 1)

    puts "################################\n"
    puts "Network output:\n#{cnn.layers.last.as(SHAInet::FullyConnectedLayer | SHAInet::SoftmaxLayer).output}\n"
    puts "Error signal is:\n#{cnn.error_signal}"

    # cnn.layers.each_with_index do |layer, i|
    #   puts "Layer #{i} - #{layer.class}:"
    #   layer.inspect("gradients")
    # end

    #
  end
  #
end
