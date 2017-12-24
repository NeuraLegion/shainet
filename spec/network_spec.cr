require "./spec_helper"

describe SHAInet::Network do
  # TODO: Write tests
  it "figure out xor" do
    training_data = [[[0, 0], [0]],
                     [[1, 0], [1]],
                     [[0, 1], [1]],
                     [[1, 1], [0]]]

    xor = SHAInet::Network.new
    xor.add_layer(:input, 2, :memory)
    1.times { |x| xor.add_layer(:hidden, 2, :memory) }
    xor.add_layer(:output, 1, :memory)
    xor.fully_connect

    # data, cost_function, activation_function, epochs, error_threshold, learning_rate, momentum)
    xor.train(training_data, :mse, :sigmoid, 10000, 0.000001)

    puts "-----------"

    (xor.run([0, 0]).first < 0.1).should eq(true)
  end
end
