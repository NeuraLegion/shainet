require "./spec_helper"

describe SHAInet::RecurrentLayer do
  it "runs forward and backward through a simple sequence" do
    net = SHAInet::Network.new
    net.add_layer(:input, 1, :memory, SHAInet.none)
    net.add_layer(:recurrent, 1, :memory, SHAInet.sigmoid)
    net.add_layer(:output, 1, :memory, SHAInet.sigmoid)
    net.fully_connect

    seq = [[1.0], [2.0], [3.0]]
    before = net.all_synapses.first.weight
    net.train([[seq, [0.5]]], training_type: :sgdm, epochs: 1, mini_batch_size: 1, log_each: 1)
    after = net.all_synapses.first.weight
    (before != after).should eq(true)
    outputs = net.run(seq)
    outputs.size.should eq(3)
  end

  it "learns to predict the last value of a sequence" do
    net = SHAInet::Network.new
    net.add_layer(:input, 1, :memory, SHAInet.none)
    net.add_layer(:recurrent, 1, :memory, SHAInet.sigmoid)
    net.add_layer(:output, 1, :memory, SHAInet.none)
    net.fully_connect

    seq = [[1.0], [2.0], [3.0]]
    expected = 3.0
    net.train([[seq, [expected]]], training_type: :sgdm,
      epochs: 500, mini_batch_size: 1, log_each: 500)
    result = net.run(seq).last.first
    result.should be_close(expected, 0.1)
  end
end
