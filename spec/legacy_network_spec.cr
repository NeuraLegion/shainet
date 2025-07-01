require "./spec_helper"

describe SHAInet::LegacyNetwork do
  it "delegates to the internal Network" do
    net = SHAInet::LegacyNetwork.new
    layer1 = net.add_layer(2, 3)
    layer2 = net.add_layer(3, 1)

    input = SHAInet::SimpleMatrix.from_a([[1.0, 2.0]])
    out = net.forward(input)
    out.rows.should eq 1
    out.cols.should eq 1

    grad_out = SHAInet::SimpleMatrix.ones(1, 1)
    grad_in = net.backward(grad_out)
    grad_in.rows.should eq 1
    grad_in.cols.should eq 2
  end
end
