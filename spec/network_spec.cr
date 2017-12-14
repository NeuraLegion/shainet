require "./spec_helper"

describe SHAInet::NeuralNet do
  # TODO: Write tests
  it "Initialize" do
    nn = SHAInet::NeuralNet.new(1, [2, 3], 5)
    nn.inspect
    nn.should be_a(SHAInet::NeuralNet)
  end

  it "Creates a fully connected network" do
    nn = SHAInet::NeuralNet.new(4, [5, 3], 5)
    nn.fully_connect
    nn.inspect
    (nn.synapses.empty?).should eq(false)
  end

  it "randomize all wights" do
    nn = SHAInet::NeuralNet.new(4, [5, 3], 5)
    nn.fully_connect
    nn.randomize_all_wights
    nn.inspect
    nn.synapses.sample.wight.should_not eq(0.0)
  end
end
