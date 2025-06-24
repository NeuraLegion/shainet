require "./spec_helper"

describe SHAInet::StreamingData do
  it "streams batches from disk during training" do
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    File.open("/tmp/stream.txt", "w") do |f|
      f.puts "[[0,0],[0]]"
      f.puts "[[1,0],[1]]"
      f.puts "[[0,1],[1]]"
      f.puts "[[1,1],[0]]"
    end

    data = SHAInet::StreamingData.new("/tmp/stream.txt")

    net = SHAInet::Network.new
    net.add_layer(:input, 2, :memory, SHAInet.sigmoid)
    net.add_layer(:hidden, 3, :memory, SHAInet.sigmoid)
    net.add_layer(:output, 1, :memory, SHAInet.sigmoid)
    net.fully_connect
    net.learning_rate = 0.7
    net.momentum = 0.3

    net.train(
      data: data,
      training_type: :sgdm,
      cost_function: :mse,
      epochs: 5000,
      mini_batch_size: 2,
      log_each: 10,
      show_slice: false)

    (net.run(input: [0, 0], stealth: true).first < 0.2).should be_true
  end
end
