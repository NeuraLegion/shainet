require "./spec_helper"
require "csv"

describe SHAInet::Network do
  puts "############################################################"

  it "Initialize" do
    puts "\n"
    nn = SHAInet::Network.new
    nn.should be_a(SHAInet::Network)
  end

  puts "############################################################"

  it "Saves to file" do
    puts "\n"
    nn = SHAInet::Network.new
    nn.add_layer(:input, 2, :memory, SHAInet.sigmoid)
    nn.add_layer(:output, 2, :memory, SHAInet.sigmoid)
    nn.add_layer(:hidden, 2, :memory, SHAInet.sigmoid)
    nn.fully_connect
    nn.save_to_file("#{__DIR__}/my_net.nn")
    File.exists?("#{__DIR__}/my_net.nn").should eq(true)
  end

  puts "############################################################"

  it "Loads from file" do
    puts "\n"
    nn = SHAInet::Network.new
    nn.load_from_file("#{__DIR__}/my_net.nn")
    (nn.all_neurons.size > 0).should eq(true)
  end

  puts "############################################################"

  it "Works on a linear regression model" do
    puts "\n"
    # data structures to hold the input and results
    inputs = Array(Array(Float64)).new
    outputs = Array(Array(Float64)).new

    # read the file
    raw = File.read("./spec/linear_data/data.csv")
    csv = CSV.new(raw, headers: true)

    # load the data structures
    while (csv.next)
      inputs << [csv.row["Height"].to_f64]
      outputs << [csv.row["Weight"].to_f64]
    end

    # normalize the data
    training = SHAInet::TrainingData.new(inputs, outputs)

    # create a network
    model = SHAInet::Network.new
    model.add_layer(:input, 1, :memory, SHAInet.none)
    # model.add_layer(:hidden, 1, :memory, SHAInet.none)
    model.add_layer(:output, 1, :memory, SHAInet.none)
    model.fully_connect

    # Update learing rate (default is 0.005)
    model.learning_rate = 0.01
    model.momentum = 0.2

    # train the network using Stochastic Gradient Descent with momentum
    model.train(data: training.raw_data,
      training_type: :sgdm,
      cost_function: :mse,
      epochs: 5000,
      error_threshold: 1e-9,
      log_each: 1000)

    # model.train_es(
    #   data: training.raw_data,
    #   pool_size: 50,
    #   learning_rate: 0.5,
    #   sigma: 0.1,
    #   cost_function: :mse,
    #   epochs: 1000,
    #   mini_batch_size: 5,
    #   error_threshold: 1e-9,
    #   log_each: 100,
    #   show_slice: true)

    # model.show

    # Test model
    output = model.run(input: [1.47], stealth: true).first
    error = ((output - 51.008)/51.008).abs
    puts "-----"
    puts "Input: \t\t1.47"
    puts "Expected: \t51.008"
    puts "Actual: \t#{output.round(3)}"
    puts "Error: \t\t#{(error*100).round(2)}%"
    (error < 0.05).should eq(true) # require less than 5% error

    output = model.run(input: [1.83], stealth: true).first
    error = ((output - 73.066)/73.066).abs
    puts "-----"
    puts "Input: \t\t1.83"
    puts "Expected: \t73.066"
    puts "Actual: \t#{output.round(3)}"
    puts "Error: \t\t#{(error*100).round(2)}%"
    (error < 0.05).should eq(true) # require less than 5% error
  end

  # puts "############################################################"

  it "Figure out XOR with SGD + M" do
    puts "\n"
    training_data = [
      [[0, 0], [0]],
      [[1, 0], [1]],
      [[0, 1], [1]],
      [[1, 1], [0]],
    ]

    xor = SHAInet::Network.new

    xor.add_layer(:input, 2, "memory", SHAInet.sigmoid)
    1.times { |x| xor.add_layer(:hidden, 3, "memory", SHAInet.sigmoid) }
    xor.add_layer(:output, 1, "memory", SHAInet.sigmoid)
    xor.fully_connect

    xor.learning_rate = 0.7
    xor.momentum = 0.3

    xor.train(
      data: training_data,
      training_type: :sgdm,
      cost_function: :mse,
      epochs: 5000,
      error_threshold: 1e-9,
      log_each: 1000)

    (xor.run(input: [0, 0], stealth: false).first < 0.1).should eq(true)
    (xor.run(input: [1, 0], stealth: false).first > 0.9).should eq(true)
    (xor.run(input: [0, 1], stealth: false).first > 0.9).should eq(true)
    (xor.run(input: [1, 1], stealth: false).first < 0.1).should eq(true)
  end

  puts "############################################################"

  it "Supports both Symbols or Strings as input params" do
    puts "\n"
    # puts "Supports both Symbols or Strings as input params (sgdm, train, mse, sigmoid)"
    training_data = [
      [[0, 0], [0]],
      [[1, 0], [1]],
      [[0, 1], [1]],
      [[1, 1], [0]],
    ]

    xor = SHAInet::Network.new
    xor.add_layer("input", 2, "memory", SHAInet.sigmoid)
    1.times { |x| xor.add_layer("hidden", 3, "memory", SHAInet.sigmoid) }
    xor.add_layer("output", 1, "memory", SHAInet.sigmoid)
    xor.fully_connect

    xor.learning_rate = 0.7
    xor.momentum = 0.3

    xor.train(
      data: training_data,
      training_type: "sgdm",
      cost_function: "mse",
      epochs: 5000,
      error_threshold: 1e-9,
      log_each: 1000)

    (xor.run(input: [0, 0], stealth: false).first < 0.1).should eq(true)
    (xor.run(input: [1, 0], stealth: false).first > 0.9).should eq(true)
    (xor.run(input: [0, 1], stealth: false).first > 0.9).should eq(true)
    (xor.run(input: [1, 1], stealth: false).first < 0.1).should eq(true)
  end

  puts "############################################################"

  # it "Works on iris dataset with batch train with Rprop (mini-batch)" do
  #   puts "\n"
  #   label = {
  #     "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
  #     "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
  #     "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
  #   }
  #   iris = SHAInet::Network.new
  #   iris.add_layer(:input, 4, :memory, SHAInet.sigmoid)
  #   iris.add_layer(:hidden, 4, :memory, SHAInet.sigmoid)
  #   iris.add_layer(:output, 3, :memory, SHAInet.sigmoid)
  #   iris.fully_connect

  #   outputs = Array(Array(Float64)).new
  #   inputs = Array(Array(Float64)).new
  #   CSV.each_row(File.read(__DIR__ + "/test_data/iris.csv")) do |row|
  #     row_arr = Array(Float64).new
  #     row[0..-2].each do |num|
  #       row_arr << num.to_f64
  #     end
  #     inputs << row_arr
  #     outputs << label[row[-1]]
  #   end

  #   data = SHAInet::TrainingData.new(inputs, outputs)
  #   data.normalize_min_max

  #   training_data, test_data = data.split(0.75) # Split also shuffles

  #   iris.train(
  #     data: training_data,
  #     training_type: :rprop,
  #     cost_function: :mse,
  #     epochs: 5000,
  #     error_threshold: 1e-9,
  #     mini_batch_size: 4,
  #     log_each: 1000,
  #     show_slice: false)

  #   # Test the trained model
  #   correct = 0
  #   test_data.data.each do |data_point|
  #     result = iris.run(data_point[0], stealth: true)
  #     expected = data_point[1]
  #     # puts "result: \t#{result.map { |x| x.round(5) }}"
  #     # puts "expected: \t#{expected}"
  #     error_sum = 0.0
  #     result.size.times do |i|
  #       error_sum += (result[i] - expected[i]).abs
  #     end
  #     correct += 1 if error_sum < 0.3
  #   end
  #   accuracy = (correct.to_f64 / test_data.size)
  #   puts "Correct answers: #{correct} / #{test_data.size}, Accuracy: #{(accuracy*100).round(3)}%"
  #   (accuracy >= 0.6).should eq(true)
  # end

  # puts "############################################################"

  it "Works on iris dataset with mini-batch train with Adam (mini-batch)" do
    puts "\n"
    label = {
      "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
      "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
      "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
    }
    iris = SHAInet::Network.new
    iris.add_layer(:input, 4, :memory, SHAInet.sigmoid)
    iris.add_layer(:hidden, 4, :memory, SHAInet.sigmoid)
    iris.add_layer(:output, 3, :memory, SHAInet.sigmoid)
    iris.fully_connect

    iris.learning_rate = 0.7
    iris.momentum = 0.3

    outputs = Array(Array(Float64)).new
    inputs = Array(Array(Float64)).new
    CSV.each_row(File.read(__DIR__ + "/test_data/iris.csv")) do |row|
      row_arr = Array(Float64).new
      row[0..-2].each do |num|
        row_arr << num.to_f64
      end
      inputs << row_arr
      outputs << label[row[-1]]
    end
    data = SHAInet::TrainingData.new(inputs, outputs)
    data.normalize_min_max

    training_data, test_data = data.split(0.75) # Split also shuffles

    iris.train(
      data: training_data,
      training_type: :adam,
      cost_function: :mse,
      epochs: 5000,
      error_threshold: 1e-9,
      mini_batch_size: 10,
      log_each: 1000,
      show_slice: false)

    # Test the trained model
    correct = 0
    test_data.data.each do |data_point|
      result = iris.run(data_point[0], stealth: true)
      expected = data_point[1]
      # puts "result: \t#{result.map { |x| x.round(5) }}"
      # puts "expected: \t#{expected}"
      error_sum = 0.0
      result.size.times do |i|
        error_sum += (result[i] - expected[i]).abs
      end
      correct += 1 if error_sum < 0.3
    end
    accuracy = (correct.to_f64 / test_data.size)
    puts "Correct answers: #{correct} / #{test_data.size}, Accuracy: #{(accuracy*100).round(3)}%"
    (accuracy >= 0.6).should eq(true)
  end

  puts "############################################################"

  it "Trains , saves, loads, runs" do
    puts "\n"
    label = {
      "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
      "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
      "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
    }
    iris = SHAInet::Network.new
    iris.add_layer(:input, 4, :memory, SHAInet.sigmoid)
    iris.add_layer(:hidden, 4, :memory, SHAInet.sigmoid)
    iris.add_layer(:output, 3, :memory, SHAInet.sigmoid)
    iris.fully_connect

    iris.learning_rate = 0.7
    iris.momentum = 0.3

    outputs = Array(Array(Float64)).new
    inputs = Array(Array(Float64)).new
    CSV.each_row(File.read(__DIR__ + "/test_data/iris.csv")) do |row|
      row_arr = Array(Float64).new
      row[0..-2].each do |num|
        row_arr << num.to_f64
      end
      inputs << row_arr
      outputs << label[row[-1]]
    end
    data = SHAInet::TrainingData.new(inputs, outputs)
    data.normalize_min_max

    training_data, test_data = data.split(0.75) # Split also shuffles

    iris.train(
      data: training_data,
      training_type: :adam,
      cost_function: :mse,
      epochs: 5000,
      error_threshold: 1e-9,
      mini_batch_size: 4,
      log_each: 1000,
      show_slice: false)

    iris.save_to_file("./my_net.nn")
    nn = SHAInet::Network.new
    nn.load_from_file("./my_net.nn")

    # Test the trained model
    correct = 0
    test_data.data.each do |data_point|
      result = iris.run(data_point[0], stealth: true)
      expected = data_point[1]
      # puts "result: \t#{result.map { |x| x.round(5) }}"
      # puts "expected: \t#{expected}"
      error_sum = 0.0
      result.size.times do |i|
        error_sum += (result[i] - expected[i]).abs
      end
      correct += 1 if error_sum < 0.3
    end
    accuracy = (correct.to_f64 / test_data.size)
    puts "Correct answers: #{correct} / #{test_data.size}, Accuracy: #{(accuracy*100).round(3)}%"
    (accuracy >= 0.6).should eq(true)
  end

  puts "############################################################"

  it "Performs autosave during training" do
    puts "\n"
    training_data = [
      [[0, 0], [0]],
      [[1, 0], [1]],
      [[0, 1], [1]],
      [[1, 1], [0]],
    ]

    xor = SHAInet::Network.new
    xor.add_layer(:input, 2, "memory", SHAInet.sigmoid)
    1.times { |x| xor.add_layer(:hidden, 3, "memory", SHAInet.sigmoid) }
    xor.add_layer(:output, 1, "memory", SHAInet.sigmoid)
    xor.fully_connect

    xor.learning_rate = 0.7
    xor.momentum = 0.3

    xor.train(
      data: training_data,
      training_type: :sgdm,
      cost_function: :mse,
      epochs: 2,
      error_threshold: 1e-9,
      log_each: 1,
      autosave: {freq: 2, path: "#{__DIR__}"})

    File.exists?("#{__DIR__}/autosave_epoch_2.nn").should eq(true)
  end

  puts "############################################################"

  it "Works with cross-entropy" do
    puts "\n"
    label = {
      "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
      "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
      "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
    }
    iris = SHAInet::Network.new
    iris.add_layer(:input, 4, :memory, SHAInet.sigmoid)
    iris.add_layer(:hidden, 4, :memory, SHAInet.sigmoid)
    iris.add_layer(:output, 3, :memory, SHAInet.sigmoid)
    iris.fully_connect

    iris.learning_rate = 0.7
    iris.momentum = 0.3

    outputs = Array(Array(Float64)).new
    inputs = Array(Array(Float64)).new
    CSV.each_row(File.read(__DIR__ + "/test_data/iris.csv")) do |row|
      row_arr = Array(Float64).new
      row[0..-2].each do |num|
        row_arr << num.to_f64
      end
      inputs << row_arr
      outputs << label[row[-1]]
    end
    data = SHAInet::TrainingData.new(inputs, outputs)
    data.normalize_min_max

    training_data, test_data = data.split(0.75) # Split also shuffles

    iris.train(
      data: training_data,
      training_type: :sgdm,
      cost_function: :c_ent,
      epochs: 5000,
      error_threshold: 1e-9,
      mini_batch_size: 4,
      log_each: 1000,
      show_slice: false)

    # Test the trained model
    correct = 0
    test_data.data.each do |data_point|
      result = iris.run(data_point[0], stealth: true)
      expected = data_point[1]
      # puts "result: \t#{result.map { |x| x.round(5) }}"
      # puts "expected: \t#{expected}"
      error_sum = 0.0
      result.size.times do |i|
        error_sum += (result[i] - expected[i]).abs
      end
      correct += 1 if error_sum < 0.3
    end
    accuracy = (correct.to_f64 / test_data.size)
    puts "Correct answers: #{correct} / #{test_data.size}, Accuracy: #{(accuracy*100).round(3)}%"
    (accuracy >= 0.6).should eq(true)
  end

  puts "############################################################"

  it "Works on iris dataset using evolutionary strategies as optimizer + cross-entropy" do
    puts "\n"
    label = {
      "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
      "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
      "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
    }

    iris = SHAInet::Network.new
    iris.add_layer(:input, 4, :memory, SHAInet.sigmoid)
    iris.add_layer(:hidden, 4, :memory, SHAInet.sigmoid)
    iris.add_layer(:output, 3, :memory, SHAInet.sigmoid)
    iris.fully_connect

    outputs = Array(Array(Float64)).new
    inputs = Array(Array(Float64)).new
    CSV.each_row(File.read(__DIR__ + "/test_data/iris.csv")) do |row|
      row_arr = Array(Float64).new
      row[0..-2].each do |num|
        row_arr << num.to_f64
      end
      inputs << row_arr
      outputs << label[row[-1]]
    end
    data = SHAInet::TrainingData.new(inputs, outputs)
    data.normalize_min_max

    training_data, test_data = data.split(0.75)

    iris.train_es(
      data: training_data,
      pool_size: 50,
      learning_rate: 0.3,
      sigma: 0.15,
      cost_function: :c_ent,
      epochs: 1000,
      mini_batch_size: 5,
      error_threshold: 1e-9,
      log_each: 10,
      show_slice: false)

    # Test the trained model
    correct = 0
    test_data.data.each do |data_point|
      result = iris.run(data_point[0], stealth: true)
      expected = data_point[1]
      # puts "result: \t#{result.map { |x| x.round(5) }}"
      # puts "expected: \t#{expected}"
      error_sum = 0.0
      result.size.times do |i|
        error_sum += (result[i] - expected[i]).abs
      end
      correct += 1 if error_sum < 0.3
    end
    accuracy = (correct.to_f64 / test_data.size)
    puts "Correct answers: #{correct} / #{test_data.size}, Accuracy: #{(accuracy*100).round(3)}%"
    (accuracy >= 0.6).should eq(true)
  end

  puts "############################################################"

  # it "Test matrix forward prop" do
  #   puts "\n"
  #   label = {
  #     "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
  #     "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
  #     "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
  #   }

  #   iris = SHAInet::Network.new
  #   iris.add_layer(:input, 4, :memory, SHAInet.none)
  #   iris.add_layer(:hidden, 4, :memory, SHAInet.none)
  #   iris.add_layer(:output, 3, :memory, SHAInet.sigmoid)
  #   iris.fully_connect

  #   outputs = Array(Array(Float64)).new
  #   inputs = Array(Array(Float64)).new
  #   CSV.each_row(File.read(__DIR__ + "/test_data/iris.csv")) do |row|
  #     row_arr = Array(Float64).new
  #     row[0..-2].each do |num|
  #       row_arr << num.to_f64
  #     end
  #     inputs << row_arr
  #     outputs << label[row[-1]]
  #   end
  #   data = SHAInet::TrainingData.new(inputs, outputs)
  #   data.normalize_min_max

  #   training_data, test_data = data.split(0.75)

  #   iris.train_es_exp(
  #     data: training_data,
  #     pool_size: 50,
  #     learning_rate: 0.5,
  #     sigma: 0.1,
  #     cost_function: :c_ent,
  #     epochs: 1000,
  #     mini_batch_size: 5,
  #     error_threshold: 1e-9,
  #     log_each: 100,
  #     show_slice: true)

  #   # Test the trained model
  #   correct = 0
  #   test_data.data.each do |data_point|
  #     result = iris.run(data_point[0], stealth: true)
  #     expected = data_point[1]
  #     # puts "result: \t#{result.map { |x| x.round(5) }}"
  #     # puts "expected: \t#{expected}"
  #     error_sum = 0.0
  #     result.size.times do |i|
  #       error_sum += (result[i] - expected[i]).abs
  #     end
  #     correct += 1 if error_sum < 0.3
  #   end
  #   accuracy = (correct.to_f64 / test_data.size)
  #   puts "Correct answers: #{correct} / #{test_data.size}, Accuracy: #{(accuracy*100).round(3)}%"
  #   (accuracy >= 0.6).should eq(true)
  # end

  # puts "############################################################"

  # # it "Works on the mnist dataset using evolutionary optimizer and batch" do
  # #   mnist = SHAInet::Network.new
  # #   mnist.add_layer(:input, 784, "memory", SHAInet.none)
  # #   mnist.add_layer(:hidden, 10, "memory", SHAInet.tanh)
  # #   # mnist.add_layer(:hidden, 40, "eraser", SHAInet.sigmoid)
  # #   mnist.add_layer(:hidden, 10, "memory", SHAInet.sigmoid)
  # #   # mnist.add_layer(:hidden, 100, "memory", SHAInet.sigmoid)
  # #   mnist.add_layer(:output, 10, "memory", SHAInet.sigmoid)

  # #   # Input to first hidden
  # #   mnist.fully_connect

  # #   # Load training data
  # #   raw_data = Array(Array(Float64)).new
  # #   csv = CSV.new(File.read(__DIR__ + "/test_data/mnist_train.csv"))
  # #   5000.times do
  # #     # CSV.each_row(File.read(__DIR__ + "/test_data/mnist_train.csv")) do |row|
  # #     csv.next
  # #     new_row = Array(Float64).new
  # #     csv.row.to_a.each { |value| new_row << value.to_f64 }
  # #     raw_data << new_row
  # #   end
  # #   raw_input_data = Array(Array(Float64)).new
  # #   raw_output_data = Array(Array(Float64)).new

  # #   raw_data.each do |row|
  # #     raw_input_data << row[1..-1]
  # #     raw_output_data << [row[0]]
  # #   end

  # #   training_data = SHAInet::TrainingData.new(raw_input_data, raw_output_data)
  # #   # training_data.normalize_min_max
  # #   training_data.normalized_inputs = training_data.normalize_min_max(data: training_data.inputs)
  # #   training_data.normalized_outputs = training_data.to_onehot(data: training_data.outputs, vector_size: 10)

  # #   # Train on the data
  # #   mnist.train_es(
  # #     data: training_data,
  # #     pool_size: 50,
  # #     learning_rate: 0.5,
  # #     sigma: 0.1,
  # #     cost_function: :c_ent,
  # #     epochs: 1,
  # #     mini_batch_size: 50,
  # #     error_threshold: 0.00000001,
  # #     log_each: 1,
  # #     show_slice: true)

  # #   # Load test data
  # #   raw_data = Array(Array(Float64)).new
  # #   csv = CSV.new(File.read(__DIR__ + "/test_data/mnist_test.csv"))
  # #   1000.times do
  # #     # CSV.each_row(File.read(__DIR__ + "/test_data/mnist_train.csv")) do |row|
  # #     csv.next
  # #     new_row = Array(Float64).new
  # #     csv.row.to_a.each { |value| new_row << value.to_f64 }
  # #     raw_data << new_row
  # #   end
  # #   raw_input_data = Array(Array(Float64)).new
  # #   raw_output_data = Array(Array(Float64)).new

  # #   raw_data.each do |row|
  # #     raw_input_data << row[1..-1]
  # #     raw_output_data << [row[0]]
  # #   end

  # #   test_data = SHAInet::TrainingData.new(raw_input_data, raw_output_data)
  # #   test_data.normalized_inputs = test_data.normalize_min_max(data: test_data.inputs)
  # #   test_data.normalized_outputs = test_data.to_onehot(data: test_data.outputs, vector_size: 10)

  # #   # Run on all test data
  # #   correct = 0
  # #   total = test_data.normalized_outputs.size

  # #   test_data.normalized_inputs.each_with_index do |test, i|
  # #     result = mnist.run(input: test, stealth: true)
  # #     result_lbl = result.index(result.max)
  # #     expected_lbl = test_data.normalized_outputs[i].index(test_data.normalized_outputs[i].max)

  # #     correct += 1 if result_lbl == expected_lbl
  # #   end
  # #   accuracy = (correct.to_f64 / total)
  # #   puts "Correct answers: #{correct} / #{total}, Accuracy: #{(accuracy*100).round(3)}%"
  # #   (accuracy >= 0.2).should eq(true) # Without training acc is ~= 10%
  # # end
end

# Remove train data
system("cd #{__DIR__} && rm *.nn")
# system("cd #{__DIR__}/test_data && rm *.csv")
File.delete("my_net.nn") rescue nil
File.delete("xor.nn") rescue nil
File.delete("autosave_epoch_2.nn") rescue nil
