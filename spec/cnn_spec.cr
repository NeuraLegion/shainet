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

  # it "Figure out MNIST (mini-batch train, mse)" do
  #   # Load training data (partial dataset)
  #   raw_data = Array(Array(Float64)).new
  #   csv = CSV.new(File.read(__DIR__ + "/test_data/mnist_train.csv"))
  #   1000.times do
  #     # CSV.each_row(File.read(__DIR__ + "/test_data/mnist_train.csv")) do |row|
  #     csv.next
  #     new_row = Array(Float64).new
  #     csv.row.to_a.each { |value| new_row << value.to_f64 }
  #     raw_data << new_row
  #   end
  #   raw_input_data = Array(Array(Float64)).new
  #   raw_output_data = Array(Array(Float64)).new

  #   raw_data.each do |row|
  #     raw_input_data << row[1..-1]
  #     raw_output_data << [row[0]]
  #   end

  #   training_data = SHAInet::CNNData.new(raw_input_data, raw_output_data)
  #   training_data.for_mnist_conv
  #   training_data.data_pairs.shuffle!

  #   # puts "#{training_data.data_pairs.first[:output]}"
  #   # training_data.data_pairs.first[:input].first.each do |row|
  #   #   puts "#{row}"
  #   # end

  #   # Load test data (partial dataset)
  #   raw_data = Array(Array(Float64)).new
  #   csv = CSV.new(File.read(__DIR__ + "/test_data/mnist_test.csv"))
  #   100.times do
  #     csv.next
  #     new_row = Array(Float64).new
  #     csv.row.to_a.each { |value| new_row << value.to_f64 }
  #     raw_data << new_row
  #   end

  #   raw_input_data = Array(Array(Float64)).new
  #   raw_output_data = Array(Array(Float64)).new

  #   raw_data.each do |row|
  #     raw_input_data << row[1..-1]
  #     raw_output_data << [row[0]]
  #   end

  #   test_data = SHAInet::CNNData.new(raw_input_data, raw_output_data)
  #   test_data.for_mnist_conv

  #   cnn = SHAInet::CNN.new
  #   cnn.add_input([height = 28, width = 28, channels = 1]) # Output shape = 28x28x1
  #   cnn.add_conv(
  #     filters_num: 10,
  #     window_size: 5,
  #     stride: 1,
  #     padding: 2,
  #     activation_function: SHAInet.none) # Output shape = 28x28x20
  #   cnn.add_relu(0.01)                   # Output shape = 28x28x20
  #   cnn.add_fconnect(l_size: 10, activation_function: SHAInet.none)
  #   cnn.add_softmax

  #   cnn.learning_rate = 0.05
  #   cnn.momentum = 0.02

  #   # cnn.run(test_data.data_pairs.first[:input], stealth = false)
  #   cnn.train_batch(
  #     data: training_data.data_pairs,
  #     training_type: :sgdm,
  #     cost_function: :c_ent,
  #     epochs: 10,
  #     error_threshold: 0.0001,
  #     log_each: 1,
  #     mini_batch_size: 50)

  #   correct_answers = 0
  #   test_data.data_pairs.each do |data_point|
  #     result = cnn.run(data_point[:input], stealth: true)
  #     if (result.index(result.max) == data_point[:output].index(data_point[:output].max))
  #       correct_answers += 1
  #     end
  #   end
  #   # cnn.inspect("activations")
  #   puts "We managed #{correct_answers} out of #{test_data.data_pairs.size} total"
  #   puts "Cnn output: #{cnn.output}"
  # end

  # it "Figure out MNIST (mini-batch train, cross-entropy)" do
  #   puts "Figure out MNIST (mini-batch train, cross-entropy)"
  #   # Load training data (partial dataset)
  #   raw_data = Array(Array(Float64)).new
  #   csv = CSV.new(File.read(__DIR__ + "/test_data/mnist_train.csv"))
  #   1000.times do
  #     # CSV.each_row(File.read(__DIR__ + "/test_data/mnist_train.csv")) do |row|
  #     csv.next
  #     new_row = Array(Float64).new
  #     csv.row.to_a.each { |value| new_row << value.to_f64 }
  #     raw_data << new_row
  #   end
  #   raw_input_data = Array(Array(Float64)).new
  #   raw_output_data = Array(Array(Float64)).new

  #   raw_data.each do |row|
  #     raw_input_data << row[1..-1]
  #     raw_output_data << [row[0]]
  #   end

  #   training_data = SHAInet::CNNData.new(raw_input_data, raw_output_data)
  #   training_data.for_mnist_conv
  #   training_data.data_pairs.shuffle!

  #   # puts "#{training_data.data_pairs.first[:output]}"
  #   # training_data.data_pairs.first[:input].first.each do |row|
  #   #   puts "#{row}"
  #   # end

  #   # Load test data (partial dataset)
  #   raw_data = Array(Array(Float64)).new
  #   csv = CSV.new(File.read(__DIR__ + "/test_data/mnist_test.csv"))
  #   100.times do
  #     csv.next
  #     new_row = Array(Float64).new
  #     csv.row.to_a.each { |value| new_row << value.to_f64 }
  #     raw_data << new_row
  #   end

  #   raw_input_data = Array(Array(Float64)).new
  #   raw_output_data = Array(Array(Float64)).new

  #   raw_data.each do |row|
  #     raw_input_data << row[1..-1]
  #     raw_output_data << [row[0]]
  #   end

  #   test_data = SHAInet::CNNData.new(raw_input_data, raw_output_data)
  #   test_data.for_mnist_conv

  #   cnn = SHAInet::CNN.new
  #   cnn.add_input([height = 28, width = 28, channels = 1]) # Output shape = 28x28x1
  #   cnn.add_conv(
  #     filters_num: 10,
  #     window_size: 5,
  #     stride: 1,
  #     padding: 2,
  #     activation_function: SHAInet.none) # Output shape = 28x28x20
  #   cnn.add_relu(0.01)                   # Output shape = 28x28x20
  #   cnn.add_fconnect(l_size: 10, activation_function: SHAInet.sigmoid)
  #   cnn.add_softmax

  #   cnn.learning_rate = 0.005
  #   cnn.momentum = 0.02

  #   # cnn.run(test_data.data_pairs.first[:input], stealth = false)

  #   cnn.train_es(
  #     data: training_data.data_pairs,
  #     pool_size: 50,
  #     learning_rate: 0.2,
  #     sigma: 0.3,
  #     cost_function: :c_ent,
  #     epochs: 10,
  #     mini_batch_size: 1,
  #     error_threshold: 0.00001,
  #     log_each: 1)

  #   correct_answers = 0
  #   test_data.data_pairs.each do |data_point|
  #     result = cnn.run(data_point[:input], stealth: true)
  #     if (result.index(result.max) == data_point[:output].index(data_point[:output].max))
  #       correct_answers += 1
  #     end
  #   end
  #   cnn.inspect("activations")
  #   puts "We managed #{correct_answers} out of #{test_data.data_pairs.size} total"
  #   puts "Cnn output: #{cnn.output}"
  # end
end

# Remove train data
system("cd #{__DIR__}/test_data && rm *.csv")
