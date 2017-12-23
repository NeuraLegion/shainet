require "./spec_helper"

describe SHAInet::Neuron do
  # TODO: Write tests
  it "check neuron creation" do
    payloads = ["abc", "12345", "!@#"]

    neuron1 = SHAInet::Neuron.new(:memory)
    neuron2 = SHAInet::Neuron.new(:eraser)

    pp neuron1
    pp neuron2
    # payloads_v.each { |x| puts x }
  end
end
