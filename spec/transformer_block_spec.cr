require "./spec_helper"

describe SHAInet::Network do
  it "adds multiple TransformerBlocks" do
    net = SHAInet::Network.new
    net.add_layer(:input, 2)
    net.add_layer(:transformer, 2, blocks: 3)
    net.transformer_layers.size.should eq(3)
  end
end
