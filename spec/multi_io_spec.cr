require "./spec_helper"

describe "multi input/output" do
  it "handles multiple input and output layers" do
    net = SHAInet::Network.new
    net.add_layer(:input, 1, :memory, SHAInet.none)
    net.add_layer(:input, 1, :memory, SHAInet.none)
    net.add_layer(:hidden, 2, :memory, SHAInet.sigmoid)
    net.add_layer(:output, 1, :memory, SHAInet.none)
    net.fully_connect

    # add a second output layer manually
    extra = SHAInet::MatrixLayer.new(2, 1) # input_size=2 (from hidden layer), output_size=1
    net.output_layers << extra
    net.connect_ltl(net.hidden_layers.last, extra, :full)

    net.randomize_all_weights
    result = net.run([0.1, 0.2])
    result.size.should eq(2)
  end
end
