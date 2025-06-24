require "./spec_helper"

describe SHAInet::Network do
  it "clips gradients when updating parameters" do
    nn = SHAInet::Network.new
    nn.add_layer(:input, 1, :memory, SHAInet.none)
    nn.add_layer(:output, 1, :memory, SHAInet.none)
    nn.fully_connect
    nn.clip_threshold = 0.5
    nn.w_gradient << 1000.0
    nn.b_gradient << 1000.0
    nn.b_gradient << 1000.0
    nn.update_weights(:sgdm)
    nn.update_biases(:sgdm)
    nn.all_synapses.first.gradient.should eq(0.5)
    nn.all_neurons.each { |n| n.gradient.should eq(0.5) }
  end
end
