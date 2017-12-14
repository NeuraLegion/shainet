require "./spec_helper"

describe SHAInet::NeuralNet do
  # TODO: Write tests
  it "Initialize" do
    nn = SHAInet::NeuralNet.new(1, [2, 3], 5)
    nn.inspect
    nn.should be_a(SHAInet::NeuralNet)
  end
end
