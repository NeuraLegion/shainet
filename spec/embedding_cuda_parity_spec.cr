require "./spec_helper"

describe "Embedding GPU parity" do
  it "matches CPU and GPU updates" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    cpu_net = SHAInet::Network.new
    cpu_net.add_layer(:input, 1, :memory, SHAInet.none)
    cpu_net.add_layer(:embedding, 2, :memory, SHAInet.none, vocab_size: 3)
    cpu_net.add_layer(:output, 1, :memory, SHAInet.none)
    cpu_net.fully_connect
    cpu_net.learning_rate = 0.1
    cpu_net.train(data: [[[1], [0.5]]], training_type: :sgdm, epochs: 1, mini_batch_size: 1, log_each: 1)
    cpu_emb = cpu_net.hidden_layers.first.as(SHAInet::EmbeddingLayer).embeddings.clone

    ENV.delete("SHAINET_DISABLE_CUDA")
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    gpu_net = SHAInet::Network.new
    gpu_net.add_layer(:input, 1, :memory, SHAInet.none)
    gpu_net.add_layer(:embedding, 2, :memory, SHAInet.none, vocab_size: 3)
    gpu_net.add_layer(:output, 1, :memory, SHAInet.none)
    gpu_net.fully_connect
    gpu_net.learning_rate = 0.1
    gpu_net.train(data: [[[1], [0.5]]], training_type: :sgdm, epochs: 1, mini_batch_size: 1, log_each: 1)
    gpu_emb = gpu_net.hidden_layers.first.as(SHAInet::EmbeddingLayer).embeddings.clone

    gpu_emb.rows.times do |r|
      gpu_emb.cols.times do |c|
        gpu_emb[r, c].should be_close(cpu_emb[r, c], 1e-6)
      end
    end
  end
end
