require "./spec_helper"

# Regression coverage for the GPU training path, which used to leave weights
# completely unchanged (flat MSE) because CPU-computed gradients and Adam
# updates were never uploaded to the device, and because the per-batch input /
# expected workspaces were handed back to the shared workspace pool while still
# in use.
#
# The global RNG is seeded in each example because MatrixLayer#random_fill!
# draws from it, and unseeded weight init would make the convergence
# thresholds below non-deterministic.
private def flatten_weights(layer : SHAInet::MatrixLayer) : Array(Float64)
  w = layer.weights
  values = Array(Float64).new(w.rows * w.cols)
  w.rows.times do |i|
    w.cols.times { |j| values << w[i, j] }
  end
  values
end

describe "training convergence" do
  it "drives XOR error down with adam" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)

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

    mse.call.should be < before * 0.5
  end

  it "drives error down for a multi-label sigmoid output layer" do
    Random::DEFAULT.new_seed(1234_u64, 5678_u64)

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

    output_layer = net.output_layers.first.as(SHAInet::MatrixLayer)
    hidden_layer = net.hidden_layers.first.as(SHAInet::MatrixLayer)
    output_before = flatten_weights(output_layer)
    hidden_before = flatten_weights(hidden_layer)
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

    # The original symptom was weights that never moved at all, so assert that
    # directly in addition to the error dropping: an error metric can shift for
    # unrelated reasons, but every trainable weight must have been updated.
    output_after = flatten_weights(output_layer)
    hidden_after = flatten_weights(hidden_layer)

    output_after.zip(output_before).count { |a, b| (a - b).abs > 1e-9 }.should eq(output_before.size)
    hidden_after.zip(hidden_before).count { |a, b| (a - b).abs > 1e-9 }.should eq(hidden_before.size)

    mse.call.should be < before * 0.7
  end
end
