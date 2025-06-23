require "./spec_helper"

describe SHAInet::EmbeddingLayer do
  it "returns consistent embeddings" do
    layer = SHAInet::EmbeddingLayer.new(4)
    first = layer.embed(1)
    second = layer.embed(1)
    first.should eq(second)
    layer.neurons.map(&.activation).should eq(first)
  end

  it "updates embeddings during training" do
    net = SHAInet::Network.new
    net.add_layer(:input, 1, :memory, SHAInet.none)
    net.add_layer(:embedding, 2, :memory, SHAInet.none)
    net.add_layer(:output, 1, :memory, SHAInet.none)
    net.fully_connect

    layer = net.hidden_layers.first.as(SHAInet::EmbeddingLayer)
    before = layer.embed(1).dup

    training = [ [[1], [0.5]] ]
    net.learning_rate = 0.1
    net.train(data: training, training_type: :sgdm, epochs: 1, mini_batch_size: 1, log_each: 1)

    after = layer.embeddings[1]
    after.should_not eq(before)
  end
end
