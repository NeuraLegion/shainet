require "./spec_helper"
require "csv"

# Extract train data
system("cd #{__DIR__}/test_data && tar xvf tests.tar.xz")

describe SHAInet::Network do
  it "Saves XOR to a file" do
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
    xor.save_to_file("./xor.nn")

    xor2 = SHAInet::Network.new
    xor2.load_from_file("./xor.nn")

    xor.run([0, 0]).first.should eq(xor2.run([0, 0]).first)
    xor.run([1, 0]).first.should eq(xor2.run([1, 0]).first)
    xor.run([0, 1]).first.should eq(xor2.run([0, 1]).first)
    xor.run([1, 1]).first.should eq(xor2.run([1, 1]).first)
  end
end
