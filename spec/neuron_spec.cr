require "./spec_helper"

describe SHAInet::Neuron do
  # TODO: Write tests
  it "check neuron creation" do
    payloads = ["abc", "12345", "!@#"]

    neuron1 = SHAInet::Neuron.new(:memory, 4)
    neuron2 = SHAInet::Neuron.new(:eraser, 4)

    pp neuron1
    pp neuron2
    # payloads_v.each { |x| puts x }
  end
end
