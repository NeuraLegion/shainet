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

      [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
    ]
    expected_output = [0, 0, 0, 0, 1]

    training_data = [img_data, expected_output]

    cnn = SHAInet::CNN.new

    cnn.add_input(volume = [7, 7, 2])
    cnn.add_conv(filters = 1, window_size = 3, stride = 1, padding = 1)
    cnn.add_relu
    cnn.add_maxpool(pool = 2, stride = 1)
    cnn.add_fconnect(l_size = 10, SHAInet.sigmoid)
    cnn.add_dropout(drop_percent = 10)
    cnn.add_fconnect(l_size = 5, SHAInet.sigmoid)
    cnn.add_softmax
    # Input layer params: volume = [width, height, channels]
    # Conv layer params: filters_num, window_size (one dimentional, i.e 2 = 2x2 window), stride, padding
    # Relu layer params: slope (default is set to 0.0, change for leaky relu)
    # Pool layer params: pool size (one dimentional, i.e 2 = 2x2 pool), stride
    # Fully conncet layer params: l_size, activation_function (default is SHAInet.sigmoid, use SHAInet.softmax when a softmax layer is needed)
    # Softmax layer: can only come after a FC layer, has the same output size as previous layer

    # Run the network with a single input
    cnn.run(img_data, stealth = true)
    cnn.layers.each { |layer| layer.inspect("activations") }
    # cnn.layers.each { |layer| puts "#{layer.class} => #{layer.next_layer.to_s}" }

    cnn.evaluate(img_data, expected_output, :mse)
    puts "-----"
    puts "Network output = #{cnn.layers.last.as(SHAInet::FullyConnectedLayer | SHAInet::SoftmaxLayer).output}"
    puts "Error signal is: #{cnn.error_signal}"

    puts "-----"
    # cnn.train(training_data, training_type = :sgdm, cost = :mse, epochs = 5000, threshold = 0.000001, log_each = 100)

    #
  end
  #
end
