require "./spec_helper"

describe SHAInet::Network do
  # TODO: Write tests
  # it "Initialize and connect" do
  #   # Initialize network requires Input_layers,hidden_layers, output_layers
  #   # New network has empty layers that need to be filled with neurons later
  #   nn = SHAInet::NeuralNet.new
  #   nn.inspect

  #   # add_layer needs: l_type, n_type, l_size, memory_size(default =1)

  #   nn.add_layer(:input, :memory, 2)
  #   3.times { |x| nn.add_layer(:hidden, :memory, 3) }
  #   nn.add_layer(:output, :memory, 2)

  #   nn.fully_connect
  #   # nn.hidden_layers.each { |layer| layer.neurons.each { |neuron| puts neuron.synapses_in } }
  # end

  # it "test run & evaluate" do
  #   p1 = [1.0, 1.0]
  #   p1.each { |i| i.to_f64 }

  #   nn = SHAInet::Network.new
  #   nn.add_layer(:input, 2, :memory)
  #   2.times { |x| nn.add_layer(:hidden, 3, :memory) }
  #   nn.add_layer(:output, 2, :memory)
  #   nn.fully_connect

  #   # # Network topology # #
  #   #  i   N N
  #   # h1  N N N
  #   # h2  N N N
  #   #  o    N
  #   # # # # # # # # # # # #

  #   nn.run(p1)

  #   # input, expected, cost_function, activation_function
  #   # puts xor.evaluate([1.0, 1.0], [0.0], :mse, :sigmoid)

  #   puts "-----------"
  #   pp nn.activations
  #   pp nn.biases
  #   pp nn.weights
  #   pp nn.error_gradient
  #   pp nn.bias_gradient
  #   pp nn.weight_gradient
  # end

  it "figure out xor" do
    training_data = [[[0, 0], [0]],
                     [[1, 0], [1]],
                     [[0, 1], [1]],
                     [[1, 1], [0]]]

    xor = SHAInet::Network.new
    xor.add_layer(:input, 2, :memory)
    2.times { |x| xor.add_layer(:hidden, 3, :memory) }
    xor.add_layer(:output, 1, :memory)
    xor.fully_connect

    # data, cost_function, activation_function, epochs, error_threshold, learning_rate, momentum)
    xor.train(training_data, :mse, :sigmoid, 10, 0.000001)

    puts "-----------"

    xor.all_neurons.each { |n| pp n.error }
  end
end
