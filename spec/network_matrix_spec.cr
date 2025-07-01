require "./spec_helper"

describe SHAInet::Network do
  it "performs forward and backward passes" do
    mat_klass = SHAInet::CUDA.available? ? SHAInet::CudaMatrix : SHAInet::SimpleMatrix

    net = SHAInet::Network.new
    l1 = net.add_layer(2, 3)
    l2 = net.add_layer(3, 1)

    l1.weights = mat_klass.from_a([
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ])
    l1.biases = mat_klass.from_a([[0.1, 0.2, 0.3]])

    l2.weights = mat_klass.from_a([
      [0.7],
      [0.8],
      [0.9],
    ])
    l2.biases = mat_klass.from_a([[0.4]])

    input = mat_klass.from_a([[1.0, 2.0]])
    out = net.forward(input)

    expected_hidden = [
      1*0.1 + 2*0.4 + 0.1,
      1*0.2 + 2*0.5 + 0.2,
      1*0.3 + 2*0.6 + 0.3,
    ]
    expected_output = [
      expected_hidden[0]*0.7 + expected_hidden[1]*0.8 + expected_hidden[2]*0.9 + 0.4,
    ]
    out[0,0].should be_close(expected_output[0], 1e-6)

    grad_out = mat_klass.ones(1,1)
    grad_in = net.backward(grad_out)

    # check gradients of last layer
    l2.g_w[0,0].should be_close(expected_hidden[0], 1e-6)
    l2.g_w[1,0].should be_close(expected_hidden[1], 1e-6)
    l2.g_w[2,0].should be_close(expected_hidden[2], 1e-6)
    l2.g_b[0,0].should be_close(1.0, 1e-6)

    # gradients for first layer
    l1.g_w[0,0].should be_close(1.0*0.7, 1e-6)
    l1.g_w[0,1].should be_close(1.0*0.8, 1e-6)
    l1.g_w[0,2].should be_close(1.0*0.9, 1e-6)
    l1.g_w[1,0].should be_close(2.0*0.7, 1e-6)
    l1.g_w[1,1].should be_close(2.0*0.8, 1e-6)
    l1.g_w[1,2].should be_close(2.0*0.9, 1e-6)
    l1.g_b[0,0].should be_close(0.7, 1e-6)
    l1.g_b[0,1].should be_close(0.8, 1e-6)
    l1.g_b[0,2].should be_close(0.9, 1e-6)

    # gradient w.r.t input
    grad_expected = [
      0.1*0.7 + 0.2*0.8 + 0.3*0.9,
      0.4*0.7 + 0.5*0.8 + 0.6*0.9,
    ]
    2.times do |j|
      grad_in[0,j].should be_close(grad_expected[j], 1e-6)
    end
  end
end
