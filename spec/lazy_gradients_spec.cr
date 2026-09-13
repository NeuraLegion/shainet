require "./spec_helper"

# Gradient buffers are the same shape as the weights, so on a large vocabulary
# they are not a rounding error: the 30B's output projection is 2048x151936, whose
# g_w alone is 1.245 GB of host RAM that pure inference never reads.
#
# The claim is therefore an ALLOCATION bound, and it is asserted directly on
# whether the buffers exist, in both directions: absent after construction and
# connection, present and correctly shaped once a backward pass has run.

describe "lazy gradient buffers" do
  it "allocates no gradient buffers when a layer is constructed" do
    layer = SHAInet::MatrixLayer.new(8, 4)
    layer.gradients_allocated?.should be_false
    # The placeholder must be genuinely empty, not merely mis-shaped.
    layer.g_w.rows.should eq(0)
    layer.g_w.cols.should eq(0)
    layer.g_b.rows.should eq(0)
    layer.g_b.cols.should eq(0)
  end

  it "still allocates none after fully_connect sizes the real weights" do
    # This is the case that matters: connect_ltl replaces the weights with their
    # real shape, and used to allocate matching gradients at the same time. On the
    # 30B that single allocation is the 1.245 GB one.
    net = SHAInet::Network.new
    net.add_layer(:input, 6)
    net.add_layer(:hidden, 5)
    net.add_layer(:output, 3)
    net.fully_connect

    ol = net.output_layers.last
    # The weights DID get their real shape, so this is not vacuous.
    ol.weights.rows.should be > 0
    ol.weights.cols.should eq(3)
    ol.gradients_allocated?.should be_false
  end

  it "materializes gradients matching the weights on the first backward pass" do
    net = SHAInet::Network.new
    net.add_layer(:input, 2)
    net.add_layer(:hidden, 3)
    net.add_layer(:output, 1)
    net.fully_connect

    net.output_layers.last.gradients_allocated?.should be_false

    net.train(
      data: [[[0.0, 0.0], [0.0]], [[1.0, 0.0], [1.0]], [[0.0, 1.0], [1.0]], [[1.0, 1.0], [0.0]]],
      training_type: :sgdm,
      cost_function: :mse,
      epochs: 5,
      log_each: 100)

    net.output_layers.last.gradients_allocated?.should be_true
    # Shape must track the WEIGHTS, not the constructor's in_size, since
    # fully_connect replaced them.
    ol = net.output_layers.last
    ol.g_w.rows.should eq(ol.weights.rows)
    ol.g_w.cols.should eq(ol.weights.cols)
    ol.g_b.rows.should eq(ol.biases.rows)
    ol.g_b.cols.should eq(ol.biases.cols)
  end

  it "accumulates real gradient values, not just an empty buffer" do
    # The allocation bound above could be satisfied by materializing a buffer and
    # never writing to it. Convergence is NOT re-tested here: the existing
    # training_convergence and transformer training specs already cover that, and
    # they are what caught this change breaking update_weights. What is asserted
    # here is this change's own contract, that the buffer materializes and receives
    # accumulated values.
    Random::DEFAULT.new_seed(42_u64, 54_u64)

    net = SHAInet::Network.new
    net.add_layer(:input, 2)
    net.add_layer(:hidden, 4, SHAInet.sigmoid)
    net.add_layer(:output, 1, SHAInet.sigmoid)
    net.fully_connect

    ol = net.output_layers.last
    ol.gradients_allocated?.should be_false

    data = [[[0.0, 0.0], [0.0]], [[1.0, 0.0], [1.0]], [[0.0, 1.0], [1.0]], [[1.0, 1.0], [0.0]]]
    net.train(data: data, training_type: :sgdm, cost_function: :mse,
      epochs: 3, error_threshold: -1.0, mini_batch_size: data.size, log_each: 10_000)

    ol.gradients_allocated?.should be_true
    gw = ol.g_w
    gw.rows.should eq(ol.weights.rows)
    gw.cols.should eq(ol.weights.cols)

    # At least one gradient entry must be non-zero: an all-zero buffer after a
    # backward pass would mean the accumulation never reached it.
    nonzero = 0
    gw.rows.times do |i|
      gw.cols.times { |j| nonzero += 1 if gw[i, j].abs > 0.0 }
    end
    nonzero.should be > 0
  end

  it "is idempotent, so repeated backward passes do not reallocate" do
    layer = SHAInet::MatrixLayer.new(5, 3)
    layer.materialize_gradients!
    first = layer.g_w
    layer.materialize_gradients!
    layer.g_w.should be(first) # same object, not a fresh allocation
  end
end
