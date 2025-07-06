require "./spec_helper"

describe SHAInet::Network do
  it "raises when label index is out of bounds" do
    net = SHAInet::Network.new
    net.add_layer(:input, 1, SHAInet.none)
    net.add_layer(:output, 2, SHAInet.sigmoid)
    net.fully_connect

    expect_raises(SHAInet::NeuralNetRunError) do
      net.evaluate_label([0], 3)
    end
  end

  it "raises when sequence label index is out of bounds" do
    net = SHAInet::Network.new
    net.add_layer(:input, 1, SHAInet.none)
    net.add_layer(:output, 2, SHAInet.sigmoid)
    net.fully_connect

    expect_raises(SHAInet::NeuralNetRunError) do
      net.evaluate_sequence_label([[0]], 3)
    end
  end
end
