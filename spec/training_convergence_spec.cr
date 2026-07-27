require "./spec_helper"

# Regression coverage for the GPU training path, which used to leave weights
# completely unchanged (flat MSE) because CPU-computed gradients and Adam
# updates were never uploaded to the device, and because the per-batch input /
# expected workspaces were handed back to the shared workspace pool while still
# in use.
describe "training convergence" do
  it "drives XOR error down with adam" do
    data = [
      [[0.0, 0.0], [0.0]],
      [[1.0, 0.0], [1.0]],
      [[0.0, 1.0], [1.0]],
      [[1.0, 1.0], [0.0]],
    ]

    net = SHAInet::Network.new
    net.add_layer(:input, 2)
    net.add_layer(:hidden, 4, SHAInet.sigmoid)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect

    mse = -> do
      total = 0.0
      data.each do |sample|
        got = net.run(sample[0], stealth: true)
        total += (got[0] - sample[1][0]) ** 2
      end
      total / data.size
    end

    before = mse.call

    net.train(
      data: data,
      training_type: :adam,
      cost_function: :mse,
      epochs: 3000,
      error_threshold: -1.0,
      mini_batch_size: data.size,
      log_each: 10_000
    )

    after = mse.call
    after.should be < before * 0.5
  end

  it "drives error down for a multi-label sigmoid output layer" do
    rng = Random.new(1234)
    pairs = Array(Array(Array(Float64))).new
    300.times do
      input = Array(Float64).new(20) { rng.rand < 0.3 ? 1.0 : 0.0 }
      target = Array(Float64).new(5, 0.0)
      target[0] = 1.0 if input[0] == 1.0
      target[3] = 1.0 if input[1] == 1.0 && input[2] == 1.0
      pairs << [input, target]
    end

    net = SHAInet::Network.new
    net.add_layer(:input, 20, SHAInet.sigmoid)
    net.add_layer(:hidden, 12, SHAInet.sigmoid)
    net.add_layer(:output, 5, SHAInet.sigmoid)
    net.fully_connect

    mse = -> do
      total = 0.0
      pairs.each do |sample|
        got = net.run(sample[0], stealth: true)
        sample[1].each_with_index { |t, i| total += (got[i] - t) ** 2 }
      end
      total / (pairs.size * 5)
    end

    before = mse.call

    net.train(
      data: pairs,
      training_type: :adam,
      cost_function: :mse,
      epochs: 30,
      error_threshold: -1.0,
      mini_batch_size: 32,
      log_each: 10_000
    )

    mse.call.should be < before * 0.7
  end
end
