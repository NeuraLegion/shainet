require "./spec_helper"

describe SHAInet::Neuron do
  it "Initialize" do
    neuron = SHAInet::Neuron.new(:memory)
    neuron.should be_a(SHAInet::Neuron)
  end
end
