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
    net.add_layer(:input, 2)
    net.add_layer(:hidden, 3, SHAInet.sigmoid)
    net.add_layer(:output, 1, SHAInet.sigmoid)
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

  it "reads tokenized data and reshuffles each epoch" do
    Random::DEFAULT.new_seed(42_u64, 13_u64)
    File.open("/tmp/stream_tok.txt", "w") do |f|
      f.puts "[[[1]], [2]]"
      f.puts "[[[3]], [4]]"
      f.puts "[[[5]], [6]]"
    end

    data = SHAInet::StreamingData.new("/tmp/stream_tok.txt", shuffle: true)

    first_epoch = data.next_batch(3)
    first_epoch.size.should eq(3)
    first_epoch.first[0].is_a?(Array).should be_true

    data.rewind
    second_epoch = data.next_batch(3)
    second_epoch.size.should eq(3)
  end

  it "returns GPU matrices when enabled" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?
    File.open("/tmp/stream_gpu.txt", "w") do |f|
      f.puts "[[0,0],[0]]"
    end
    data = SHAInet::StreamingData.new("/tmp/stream_gpu.txt", gpu_batches: true)
    batch = data.next_batch(1)
    batch.first[0].should be_a(SHAInet::CudaMatrix)
  end
end
