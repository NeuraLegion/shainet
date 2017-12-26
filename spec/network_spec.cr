require "./spec_helper"
require "csv"

describe SHAInet::Network do
  # it "Initialize" do
  #   nn = SHAInet::Network.new
  #   nn.should be_a(SHAInet::Network)
  # end

  # it "figure out xor" do
  #   training_data = [
  #     [[0, 0], [0]],
  #     [[1, 0], [1]],
  #     [[0, 1], [1]],
  #     [[1, 1], [0]],
  #   ]

  #   xor = SHAInet::Network.new
  #   xor.add_layer(:input, 2, :memory)
  #   1.times { |x| xor.add_layer(:hidden, 3, :memory) }
  #   xor.add_layer(:output, 1, :memory)
  #   xor.fully_connect

  #   # data, training_type, cost_function, activation_function, epochs, error_threshold (sum of errors), learning_rate, momentum)
  #   xor.train(training_data, :sgdm, :mse, :sigmoid, 10000, 0.001)

  #   (xor.run([0, 0]).first < 0.1).should eq(true)
  #   (xor.run([1, 0]).first > 0.9).should eq(true)
  #   (xor.run([0, 1]).first > 0.9).should eq(true)
  #   (xor.run([1, 1]).first < 0.1).should eq(true)
  # end

  # it "works on iris dataset" do
  #   label = {
  #     "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
  #     "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
  #     "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
  #   }
  #   iris = SHAInet::Network.new
  #   iris.add_layer(:input, 4, :memory)
  #   iris.add_layer(:hidden, 5, :memory)
  #   iris.add_layer(:output, 3, :memory)
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
  #   normalized = SHAInet::TrainingData.new(inputs, outputs)
  #   normalized.normalize_min_max
  #   puts normalized
  #   iris.train(normalized.data, :sgdm, :mse, :sigmoid, 20000, 0.1)
  #   iris.run(normalized.normalized_inputs.first)
  #   puts "Expected output is: [0,0,1]"
  # end

  # it "works on iris dataset with batch train with Rprop" do
  #   label = {
  #     "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
  #     "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
  #     "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
  #   }
  #   iris = SHAInet::Network.new
  #   iris.add_layer(:input, 4, :memory)
  #   iris.add_layer(:hidden, 5, :memory)
  #   iris.add_layer(:output, 3, :memory)
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
  #   normalized = SHAInet::TrainingData.new(inputs, outputs)
  #   normalized.normalize_min_max
  #   iris.train_batch(normalized.data, :rprop, :mse, :sigmoid, 20000, 0.01)
  #   result = iris.run(normalized.normalized_inputs.first)
  #   ((result.first < 0.3) && (result[1] < 0.3) && (result.last > 0.7)).should eq(true)
  # end

  it "works on iris dataset with batch train with Adam" do
    label = {
      "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
      "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
      "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
    }
    iris = SHAInet::Network.new
    iris.add_layer(:input, 4, :memory)
    iris.add_layer(:hidden, 5, :memory)
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
    iris.train_batch(normalized.data, :adam, :mse, :sigmoid, 20000, 0.01)
    result = iris.run(normalized.normalized_inputs.first)
    ((result.first < 0.3) && (result[1] < 0.3) && (result.last > 0.7)).should eq(true)
  end
end
