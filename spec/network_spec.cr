require "./spec_helper"
require "csv"

# Extract train data
system("cd #{__DIR__}/test_data && tar xvf tests.tar.gz")

describe SHAInet::Network do
  it "Initialize" do
    nn = SHAInet::Network.new
    nn.should be_a(SHAInet::Network)
  end

  it "Figure out XOR with SGD + M" do
    training_data = [
      [[0, 0], [0]],
      [[1, 0], [1]],
      [[0, 1], [1]],
      [[1, 1], [0]],
    ]

    xor = SHAInet::Network.new
    xor.add_layer(:input, 2, :memory)
    1.times { |x| xor.add_layer(:hidden, 3, :memory) }
    xor.add_layer(:output, 1, :memory)
    xor.fully_connect
    # data, training_type, cost_function, activation_function, epochs, error_threshold (MSE %), log each steps
    xor.train(training_data, :sgdm, :mse, :sigmoid, epochs = 5000, threshold = 0.000001, log = 100)

    (xor.run([0, 0]).first < 0.1).should eq(true)
    (xor.run([1, 0]).first > 0.9).should eq(true)
    (xor.run([0, 1]).first > 0.9).should eq(true)
    (xor.run([1, 1]).first < 0.1).should eq(true)
  end

  it "Figure out iris with SGD + M, using cross-entropy cost (no batch)" do
    label = {
      "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
      "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
      "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
    }
    iris = SHAInet::Network.new
    iris.add_layer(:input, 4, :memory)
    iris.add_layer(:hidden, 4, :memory)
    iris.add_layer(:output, 3, :memory)
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
    normalized = SHAInet::TrainingData.new(inputs, outputs)
    normalized.normalize_min_max
    iris.learning_rate = 0.7
    iris.momentum = 0.6
    iris.train(normalized.data, :sgdm, :c_ent, :sigmoid, epochs = 20000, threshold = 0.000001, log = 1000)
    result = iris.run(normalized.normalized_inputs.first)
    ((result.first < 0.3) && (result[1] < 0.3) && (result.last > 0.7)).should eq(true)
  end

  it "works on iris dataset with batch train with Rprop" do
    label = {
      "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
      "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
      "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
    }
    iris = SHAInet::Network.new
    iris.add_layer(:input, 4, :memory)
    iris.add_layer(:hidden, 4, :memory)
    iris.add_layer(:output, 3, :memory)
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
    normalized = SHAInet::TrainingData.new(inputs, outputs)
    normalized.normalize_min_max
    p "Figure out iris with iRprop+ (batch)"
    iris.train_batch(normalized.data, :rprop, :mse, :sigmoid, 5000, 0.00001)
    result = iris.run(normalized.normalized_inputs.first)
    ((result.first < 0.3) && (result[1] < 0.3) && (result.last > 0.7)).should eq(true)
  end

  it "works on iris dataset with batch train with Adam (batch)" do
    label = {
      "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
      "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
      "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
    }
    iris = SHAInet::Network.new
    iris.add_layer(:input, 4, :memory)
    iris.add_layer(:hidden, 4, :memory)
    iris.add_layer(:output, 3, :memory)
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
    normalized = SHAInet::TrainingData.new(inputs, outputs)
    normalized.normalize_min_max
    iris.train_batch(normalized.data, :adam, :mse, :sigmoid, 20000, 0.00001)
    result = iris.run(normalized.normalized_inputs.first)
    ((result.first < 0.3) && (result[1] < 0.3) && (result.last > 0.9)).should eq(true)
  end

  it "works on the mnist dataset using adam and batch" do
    mnist = SHAInet::Network.new
    mnist.add_layer(:input, 784, :memory)
    mnist.add_layer(:hidden, 20, :memory)
    mnist.add_layer(:hidden, 20, :memory)
    mnist.add_layer(:output, 1, :memory)
    mnist.fully_connect

    # Lload train data
    outputs = Array(Array(Float64)).new
    inputs = Array(Array(Float64)).new
    CSV.each_row(File.read(__DIR__ + "/test_data/mnist_train.csv")) do |row|
      row_arr = Array(Float64).new
      row[1..-1].each do |num|
        row_arr << num.to_f64
      end
      inputs << row_arr
      outputs << [row[0].to_f64]
    end
    normalized = SHAInet::TrainingData.new(inputs, outputs)
    normalized.normalize_min_max
    # Train on the data
    mnist.train_batch(normalized.data, :rprop, :mse, :sigmoid, 10, 0.0035, 10, 1000)
    mnist.train_batch(normalized.data, :adam, :mse, :sigmoid, 20000, 0.0035, 100, 1000)

    # Load test data
    outputs = Array(Array(Float64)).new
    inputs = Array(Array(Float64)).new
    results = Array(Int32).new
    CSV.each_row(File.read(__DIR__ + "/test_data/mnist_test.csv")) do |row|
      row_arr = Array(Float64).new
      row[1..-1].each do |num|
        row_arr << num.to_f64
      end
      inputs << row_arr
      outputs << [row[0].to_f64]
    end
    normalized = SHAInet::TrainingData.new(inputs, outputs)
    normalized.normalize_min_max
    # Run on all test data, and see that we are atleast 0.01 far from the right solution
    normalized.normalized_inputs.each_with_index do |test, i|
      result = mnist.run(test, :sigmoid, stealth = true)
      if (result.first - normalized.normalized_outputs[i].first).abs <= 0.01
        results << 1
      else
        results << 0
      end
    end
    puts "We managed #{results.size / results.sum}% success"
  end
end
# Remove train data
system("cd #{__DIR__}/test_data && rm *.csv")
