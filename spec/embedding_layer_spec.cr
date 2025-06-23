require "./spec_helper"

describe SHAInet::EmbeddingLayer do
  it "returns consistent embeddings" do
    layer = SHAInet::EmbeddingLayer.new(4)
    first = layer.embed(1)
    second = layer.embed(1)
    first.should eq(second)
    layer.neurons.map(&.activation).should eq(first)
  end
end
