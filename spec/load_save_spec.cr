require "./spec_helper"
require "csv"
require "json"

# Extract train data
system("cd #{__DIR__}/test_data && tar xvf tests.tar.xz")

describe SHAInet::Network do
  puts "############################################################"
  it "Saves XOR to a file" do
    puts "\n"
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
    puts "Before save"
    File.write("./xor.nn", xor.to_json)
    puts "Saved"

    xor2 = SHAInet::Network.from_json("./xor.nn")
    puts "Loaded"
    xor.run([0, 0]).first.to_f32.should eq xor2.run([0, 0]).first.to_f32
    xor.run([1, 0]).first.to_f32.should eq xor2.run([1, 0]).first.to_f32
    xor.run([0, 1]).first.to_f32.should eq xor2.run([0, 1]).first.to_f32
    xor.run([1, 1]).first.to_f32.should eq xor2.run([1, 1]).first.to_f32
  end
end
