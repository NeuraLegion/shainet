require "./spec_helper"

describe SHAInet::NeuralNet do
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

  it "figure out xor" do
    # This is testing to see if it works
    xor = SHAInet::NeuralNet.new
    xor.add_layer(:input, :memory, 2)
    3.times { |x| xor.add_layer(:hidden, :memory, 2) }
    xor.add_layer(:output, :memory, 1)

    # xor.hidden_layers.first.random_seed

    xor.fully_connect
    pp xor.hidden_layers.first.neurons

    puts xor.evaluate([1.0, 1.0])

    pp xor.hidden_layers.first.neurons

    # 10000.times do
    #   xor.train([0, 0], [0])
    #   xor.train([1, 0], [1])
    #   xor.train([0, 1], [1])
    #   xor.train([1, 1], [0])
    # end

    # xor.feed_forward([0, 0])
    # (xor.current_outputs.first < 0.1 && xor.current_outputs.first > -0.1).should eq(true)
  end
end
