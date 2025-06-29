require "./spec_helper"

private def run_serial(net : SHAInet::Network, input : Array(Float64))
  index = 0
  net.input_layers.each do |layer|
    layer.neurons.each do |neuron|
      neuron.activation = input[index]
      index += 1
    end
  end
  net.hidden_layers.each do |l|
    l.neurons.each { |n| n.activate(l.activation_function) }
  end
  net.output_layers.each do |l|
    l.neurons.each { |n| n.activate(l.activation_function) }
  end
  out = [] of Float64
  net.output_layers.each do |l|
    l.neurons.each { |n| out << n.activation }
  end
  out
end

describe "Dense matrix parity" do
  it "matches outputs of serial computation" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    net_serial = SHAInet::Network.new
    net_serial.add_layer(:input, 2)
    net_serial.add_layer(:hidden, 3)
    net_serial.add_layer(:output, 1)
    net_serial.fully_connect

    Random::DEFAULT.new_seed(42_u64, 54_u64)
    net_matrix = SHAInet::Network.new
    net_matrix.add_layer(:input, 2)
    net_matrix.add_layer(:hidden, 3)
    net_matrix.add_layer(:output, 1)
    net_matrix.fully_connect

    input = [0.2, -0.1]
    expected = run_serial(net_serial, input)
    output = net_matrix.run(input)
    output.size.should eq expected.size
    output.each_with_index do |v, i|
      v.should be_close(expected[i], 1e-6)
    end
  end
end
