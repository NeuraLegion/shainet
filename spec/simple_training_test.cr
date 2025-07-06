require "./spec_helper"

describe "Simple Matrix Training Test" do
  it "can train a tiny network on tiny data quickly" do
    puts "Testing basic matrix-based training..."

    # Create minimal network: 2 inputs -> 2 hidden -> 1 output
    net = SHAInet::Network.new
    net.add_layer(:input, 2, SHAInet.sigmoid)
    net.add_layer(:hidden, 2, SHAInet.sigmoid)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect

    # Tiny training data: XOR-like problem
    training_data = [
      [[0.0, 0.0], [0.0]],
      [[0.0, 1.0], [1.0]],
      [[1.0, 0.0], [1.0]],
      [[1.0, 1.0], [0.0]],
    ]

    puts "Network created, starting training..."
    start_time = Time.monotonic

    # Train for just a few epochs
    net.train(
      data: training_data,
      training_type: :adam,
      cost_function: :mse,
      epochs: 10, # Very small number
      error_threshold: 0.1,
      mini_batch_size: 2,
      log_each: 2,
      show_slice: true
    )

    end_time = Time.monotonic
    duration = end_time - start_time
    puts "Training completed in #{duration.total_seconds} seconds"

    # Should complete in under 10 seconds for this tiny problem
    duration.total_seconds.should be < 10.0

    # Test a simple forward pass
    result = net.run([0.5, 0.5])
    puts "Forward pass result: #{result}"
    result.size.should eq(1)
  end
end
