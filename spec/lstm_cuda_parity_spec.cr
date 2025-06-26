require "./spec_helper"

describe "LSTM GPU parity" do
  it "produces same result as CPU" do
    seq = [[1.0], [2.0], [3.0]]

    Random::DEFAULT.new_seed(42_u64, 54_u64)
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    cpu_net = SHAInet::Network.new
    cpu_net.add_layer(:input, 1, :memory, SHAInet.none)
    cpu_net.add_layer(:lstm, 1)
    cpu_net.add_layer(:output, 1, :memory, SHAInet.none)
    cpu_net.fully_connect
    cpu_net.learning_rate = 0.1
    cpu_net.train([[seq, [0.5]]], training_type: :sgdm, epochs: 1, mini_batch_size: 1, log_each: 1)
    cpu_out = cpu_net.run(seq).last.first

    ENV.delete("SHAINET_DISABLE_CUDA")
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    gpu_net = SHAInet::Network.new
    gpu_net.add_layer(:input, 1, :memory, SHAInet.none)
    gpu_net.add_layer(:lstm, 1)
    gpu_net.add_layer(:output, 1, :memory, SHAInet.none)
    gpu_net.fully_connect
    gpu_net.learning_rate = 0.1
    gpu_net.train([[seq, [0.5]]], training_type: :sgdm, epochs: 1, mini_batch_size: 1, log_each: 1)
    gpu_out = gpu_net.run(seq).last.first

    gpu_out.should be_close(cpu_out, 1e-3)
  end
end
