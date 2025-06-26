require "./spec_helper"

describe "CUDA training" do
  it "uses GPU matrices when available" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    net = SHAInet::Network.new
    net.add_layer(:input, 1, :memory, SHAInet.none)
    net.add_layer(:embedding, 2, :memory, SHAInet.none, vocab_size: 3)
    net.add_layer(:output, 1, :memory, SHAInet.none)
    net.fully_connect

    training = [[[1], [0.5]]]
    net.learning_rate = 0.1
    net.train(data: training, training_type: :sgdm, epochs: 1, mini_batch_size: 1, log_each: 1)

    layer = net.hidden_layers.first.as(SHAInet::EmbeddingLayer)
    if SHAInet::CUDA.available?
      layer.embeddings.should be_a(SHAInet::CudaMatrix)
    else
      layer.embeddings.should be_a(SHAInet::SimpleMatrix)
    end
  end
end
