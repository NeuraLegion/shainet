require "./spec_helper"
require "csv"

# Extract train data
system("cd #{__DIR__}/test_data && tar xvf tests.tar.xz")

describe SHAInet::Network do
  it "Initialize" do
    nn = SHAInet::Network.new
    nn.should be_a(SHAInet::Network)
  end

  it "saves_to_file" do
    nn = SHAInet::Network.new
    nn.add_layer(:input, 2, :memory, SHAInet.sigmoid)
    nn.add_layer(:output, 2, :memory, SHAInet.sigmoid)
    nn.add_layer(:hidden, 2, :memory, SHAInet.sigmoid)
    nn.fully_connect
    nn.save_to_file("./my_net.nn")
    File.exists?("./my_net.nn").should eq(true)
  end

  it "loads_from_file" do
    nn = SHAInet::Network.new
    nn.load_from_file("./my_net.nn")
    (nn.all_neurons.size > 0).should eq(true)
  end

  it "Figure out XOR with SGD + M" do
    training_data = [
      [[0, 0], [0]],
      [[1, 0], [1]],
      [[0, 1], [1]],
      [[1, 1], [0]],
    ]

    xor = SHAInet::Network.new
    xor.add_layer(:input, 2, :memory, SHAInet.sigmoid)
    1.times { |x| xor.add_layer(:hidden, 3, :memory, SHAInet.sigmoid) }
    xor.add_layer(:output, 1, :memory, SHAInet.sigmoid)
    xor.fully_connect
    # data, training_type, cost_function, activation_function, epochs, error_threshold (MSE %), log each steps
    xor.train(training_data, :sgdm, :mse, epochs = 5000, threshold = 0.000001, log = 100)

    (xor.run([0, 0]).first < 0.1).should eq(true)
    (xor.run([1, 0]).first > 0.9).should eq(true)
    (xor.run([0, 1]).first > 0.9).should eq(true)
    (xor.run([1, 1]).first < 0.1).should eq(true)
  end

  it "Supports both Symbols or Strings as input params" do
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
    xor.train(training_data, "sgdm", "mse", epochs = 5000, threshold = 0.000001, log = 100)

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
    normalized = SHAInet::TrainingData.new(inputs, outputs)
    normalized.normalize_min_max
    iris.learning_rate = 0.7
    iris.momentum = 0.6
    iris.train(normalized.data.shuffle, :sgdm, :c_ent, epochs = 20000, threshold = 0.000001, log = 1000)
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
    normalized = SHAInet::TrainingData.new(inputs, outputs)
    normalized.normalize_min_max
    p "Figure out iris with iRprop+ (batch)"
    iris.train_batch(normalized.data.shuffle, :rprop, :mse, 5000, 0.00001)
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
    normalized = SHAInet::TrainingData.new(inputs, outputs)
    normalized.normalize_min_max
    iris.train_batch(normalized.data.shuffle, :adam, :mse, 20000, 0.00001)
    result = iris.run(normalized.normalized_inputs.first)
    ((result.first < 0.3) && (result[1] < 0.3) && (result.last > 0.9)).should eq(true)
  end

  it "works on iris dataset with mini-batch train with Adam (mini-batch)" do
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
    normalized = SHAInet::TrainingData.new(inputs, outputs)
    normalized.normalize_min_max
    iris.train_batch(normalized.data.shuffle, :adam, :mse, 20000, 0.00001, 1000, 50)
    result = iris.run(normalized.normalized_inputs.first)
    ((result.first < 0.3) && (result[1] < 0.3) && (result.last > 0.9)).should eq(true)
  end

  it "trains , saves, loads, runs" do
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
    normalized = SHAInet::TrainingData.new(inputs, outputs)
    normalized.normalize_min_max
    iris.train_batch(normalized.data.shuffle, :adam, :mse, 20000, 0.00001, 1000, 50)

    iris.save_to_file("./my_net.nn")
    nn = SHAInet::Network.new
    nn.load_from_file("./my_net.nn")
    result = nn.run(normalized.normalized_inputs.first)
    ((result.first < 0.3) && (result[1] < 0.3) && (result.last > 0.9)).should eq(true)
  end

  # it "works on the mnist dataset using adam and batch" do
  #   mnist = SHAInet::Network.new
  #   mnist.add_layer(:input, 784, "memory", SHAInet.sigmoid)
  #   mnist.add_layer(:hidden, 100, "memory", SHAInet.sigmoid)
  #   mnist.add_layer(:hidden, 40, "eraser", SHAInet.sigmoid)
  #   mnist.add_layer(:hidden, 40, "memory", SHAInet.sigmoid)
  #   mnist.add_layer(:hidden, 100, "memory", SHAInet.sigmoid)
  #   mnist.add_layer(:output, 10, "memory", SHAInet.sigmoid)

  #   # Input to first hidden
  #   mnist.connect_ltl(mnist.input_layers.first, mnist.hidden_layers.first, :full)

  #   # first hidden to [1] and [2]
  #   mnist.connect_ltl(mnist.hidden_layers.first, mnist.hidden_layers[1], :full)
  #   mnist.connect_ltl(mnist.hidden_layers.first, mnist.hidden_layers[2], :full)

  #   # [1] and [2] to last hidden
  #   mnist.connect_ltl(mnist.hidden_layers[1], mnist.hidden_layers.last, :full)
  #   mnist.connect_ltl(mnist.hidden_layers[2], mnist.hidden_layers.last, :full)

  #   # [0] & [3] to output
  #   mnist.connect_ltl(mnist.hidden_layers[0], mnist.output_layers.first, :full)
  #   mnist.connect_ltl(mnist.hidden_layers[3], mnist.output_layers.first, :full)

  #   # Load train data
  #   outputs = Array(Array(Float64)).new
  #   inputs = Array(Array(Float64)).new
  #   CSV.each_row(File.read(__DIR__ + "/test_data/mnist_train.csv")) do |row|
  #     row_arr = Array(Float64).new
  #     row[1..-1].each do |num|
  #       row_arr << num.to_f64
  #     end
  #     inputs << row_arr
  #     a = Array(Float64).new(10, 0.0)
  #     a[row[0].to_i] = 1.0
  #     outputs << a
  #   end
  #   normalized = SHAInet::TrainingData.new(inputs, outputs)
  #   normalized.normalize_min_max
  #   # Train on the data
  #   mnist.train_batch(normalized.data.shuffle, :adam, :mse, 100, 0.0035, 10, 10000)

  #   # Load test data
  #   outputs = Array(Array(Float64)).new
  #   inputs = Array(Array(Float64)).new
  #   results = Array(Int32).new
  #   CSV.each_row(File.read(__DIR__ + "/test_data/mnist_test.csv")) do |row|
  #     row_arr = Array(Float64).new
  #     row[1..-1].each do |num|
  #       row_arr << num.to_f64
  #     end
  #     inputs << row_arr
  #     a = Array(Float64).new(10, 0.0)
  #     a[row[0].to_i] = 1.0
  #     outputs << a
  #   end
  #   normalized = SHAInet::TrainingData.new(inputs, outputs)
  #   normalized.normalize_min_max
  #   # Run on all test data, and see that we are atleast 0.01 far from the right solution
  #   normalized.normalized_inputs.each_with_index do |test, i|
  #     result = mnist.run(test, stealth = true)
  #     if (result.index(result.max) == normalized.normalized_outputs[i].index(normalized.normalized_outputs[i].max))
  #       results << 1
  #     else
  #       results << 0
  #     end
  #   end
  #   puts "We managed #{results.sum} out of #{results.size} total"
  # end
end
# Remove train data
system("cd #{__DIR__}/test_data && rm *.csv")
