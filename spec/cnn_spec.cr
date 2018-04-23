require "./spec_helper"
require "csv"

# Extract train data
system("cd #{__DIR__}/test_data && tar xvf tests.tar.xz")

describe SHAInet::CNN do
  # it "Check basic cnn features" do
  #   img_data = [
  #     [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
  #      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
  #      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
  #      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
  #      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
  #      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
  #      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],

  #     [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  #      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
  #      [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
  #      [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
  #      [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
  #      [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
  #      [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]],
  #   ]
  #   expected_output = [0.0, 0.0, 0.0, 0.0, 1.0]

  #   training_data = [[img_data, expected_output]]

  #   cnn = SHAInet::CNN.new

  #   cnn.add_input(volume = [height = 7, width = 7, channels = 2])
  #   cnn.add_conv(filters = 2, window_size = 3, stride = 1, padding = 1, activation_function = SHAInet.none)
  #   cnn.add_conv(filters = 2, window_size = 3, stride = 1, padding = 1, activation_function = SHAInet.sigmoid)
  #   cnn.add_relu
  #   cnn.add_maxpool(pool = 2, stride = 1)
  #   cnn.add_fconnect(l_size = 10, SHAInet.sigmoid)
  #   cnn.add_dropout(drop_percent = 10)
  #   cnn.add_fconnect(l_size = 5, SHAInet.none)
  #   cnn.add_softmax
  #   # ########################################################################################################################################## #
  #   # Layer parameters:
  #   # -----------------
  #   # Input layer params: volume = [width, height, channels]
  #   # Conv layer params: filters_num, window_size (one dimentional, i.e 2 = 2x2 window), stride, padding
  #   # Relu layer params: slope (default is set to 0.0, change for leaky relu)
  #   # Pool layer params: pool size (one dimentional, i.e 2 = 2x2 pool), stride
  #   # Fully conncet layer params: l_size, activation_function (default is SHAInet.sigmoid, use SHAInet.softmax when a softmax layer is needed)
  #   # Softmax layer: can only come after a FC layer, has the same output size as previous layer
  #   # ########################################################################################################################################## #

  #   # Run the network with a single input
  #   cnn.run(img_data, stealth = true)

  #   puts "################################\n"
  #   # puts "works on iris dataset with SGD+M (no batch, mse)"
  #   puts "Train using SGD+M (no batch, mse)"
  #   cnn.train(training_data, training_type = :sgdm, cost = :mse, epochs = 20000, threshold = 0.000001, log_each = 1000)

  #   puts "Network output:\n#{cnn.layers.last.as(SHAInet::FullyConnectedLayer | SHAInet::SoftmaxLayer).output}\n"
  #   puts "Error signal is:\n#{cnn.error_signal}"

  #   puts "################################\n"
  #   puts "Train using iRprop+ (mini-batch train, mse)"
  #   cnn.train_batch(training_data, training_type = :rprop, cost = :mse, epochs = 20000, threshold = 0.000001, log_each = 1000, minib = 1)

  #   puts "Network output:\n#{cnn.layers.last.as(SHAInet::FullyConnectedLayer | SHAInet::SoftmaxLayer).output}\n"
  #   puts "Error signal is:\n#{cnn.error_signal}"

  #   # cnn.layers.each_with_index do |layer, i|
  #   #   puts "Layer #{i} - #{layer.class}:"
  #   #   layer.inspect("gradients")
  #   # end

  #   #
  # end
  #

  it "Figure out MNIST (mini-batch train, mse)" do
    raw_training_data = Array(Array(Float64)).new
    raw_test_data = Array(Array(Float64)).new

    # Load training data (partial dataset)
    csv = CSV.new(File.read(__DIR__ + "/test_data/mnist_train.csv"))
    1000.times do
      # CSV.each_row(File.read(__DIR__ + "/test_data/mnist_train.csv")) do |row|
      csv.next
      new_row = Array(Float64).new
      csv.row.to_a.each { |value| new_row << value.to_f64 }
      raw_training_data << new_row
    end
    training_data = SHAInet::TrainingData.new(raw_training_data)
    training_data.for_mnist_conv
    training_data.data_pairs.shuffle!

    # Load test data (partial dataset)
    csv = CSV.new(File.read(__DIR__ + "/test_data/mnist_test.csv"))
    100.times do
      csv.next
      new_row = Array(Float64).new
      csv.row.to_a.each { |value| new_row << value.to_f64 }
      raw_test_data << new_row
    end
    test_data = SHAInet::TrainingData.new(raw_test_data)
    test_data.for_mnist_conv

    cnn = SHAInet::CNN.new
    cnn.add_input(volume = [height = 28, width = 28, channels = 1])                                          # Data = 28x28x1
    cnn.add_conv(filters = 10, window_size = 5, stride = 1, padding = 2, activation_function = SHAInet.none) # Data = 28x28x20
    cnn.add_relu                                                                                             # Data = 28x28x20
    cnn.add_maxpool(pool = 2, stride = 2)                                                                    # Data = 14x14x20
    cnn.add_conv(filters = 20, window_size = 5, stride = 1, padding = 1, activation_function = SHAInet.none) # Data = 14x14x40
    # cnn.add_maxpool(pool = 2, stride = 2)                                                                    # Data = 7x7x40
    cnn.add_fconnect(l_size = 10, activation_function = SHAInet.sigmoid)
    cnn.add_fconnect(l_size = 10, activation_function = SHAInet.sigmoid)
    cnn.add_softmax
    cnn.learning_rate = 0.5
    cnn.momentum = 0.2

    # cnn.run(test, stealth = false)
    # cnn.train(training_data.data_pairs, training_type = :sgdm, cost = :mse, epochs = 20000, threshold = 0.000001, log_each = 1000)
    cnn.train_batch(training_data.data_pairs,
      training_type = :sgdm,
      cost = :mse,
      epochs = 5,
      threshold = 0.0001,
      log_each = 1,
      minib = 25)

    correct_answers = 0
    test_data.data_pairs.each do |data_point|
      input_data = data_point[0].as(Array(Array(Array(Float64))))
      expected_output = data_point[1].as(Array(Float64))
      result = cnn.run(input_data, stealth: true)
      if (result.index(result.max) == expected_output.index(expected_output.max))
        correct_answers += 1
      end
    end
    cnn.inspect("activations")
    puts "We managed #{correct_answers} out of #{test_data.data_pairs.size} total"
    puts "Cnn output: #{cnn.output}"
  end
end

# Remove train data
system("cd #{__DIR__}/test_data && rm *.csv")
