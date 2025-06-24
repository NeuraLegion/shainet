require "./spec_helper"

describe SHAInet::MultiHeadAttention do
  it "trains to output constant values" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    attn = SHAInet::MultiHeadAttention.new(2, 1)
    input = SHAInet::SimpleMatrix.from_a([[1.0, 0.0], [0.0, 1.0]])
    target = SHAInet::SimpleMatrix.ones(2, 2)
    1000.times do
      out = attn.forward(input)
      diff = out - target
      attn.backward(diff)
      attn.apply_gradients(0.05)
    end
    out = attn.forward(input)
    out[0, 0].should be_close(1.0, 0.1)
    out[1, 1].should be_close(1.0, 0.1)
  end
end

describe SHAInet::PositionalEncoding do
  it "generates sinusoidal values" do
    pe = SHAInet::PositionalEncoding.sinusoidal(3, 4)
    pe.rows.should eq(3)
    pe.cols.should eq(4)
    pe[0, 0].should be_close(0.0, 0.0001)
    pe[0, 1].should be_close(1.0, 0.0001)
    pe[1, 0].should be_close(Math.sin(1.0), 0.0001)
    pe[1, 1].should be_close(Math.cos(1.0), 0.0001)
  end
end

describe SHAInet::TransformerLayer do
  it "overfits a tiny sequence" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    layer = SHAInet::TransformerLayer.new(2, 1, 4)
    input = SHAInet::SimpleMatrix.from_a([[1.0, 0.0], [0.0, 1.0]])
    target = SHAInet::SimpleMatrix.ones(2, 2)
    1000.times do
      out = layer.forward(input)
      diff = out - target
      layer.backward(diff)
      layer.apply_gradients(0.05)
    end
    out = layer.forward(input)
    out[0, 0].should be_close(1.0, 0.1)
    out[1, 1].should be_close(1.0, 0.1)
  end

  it "overfits with positional encoding" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    layer = SHAInet::TransformerLayer.new(2, 1, 4)
    input = SHAInet::SimpleMatrix.from_a([[1.0, 0.0], [0.0, 1.0]])
    layer.positional_encoding = SHAInet::PositionalEncoding.sinusoidal(2, 2)
    target = SHAInet::SimpleMatrix.ones(2, 2)
    1000.times do
      out = layer.forward(input)
      diff = out - target
      layer.backward(diff)
      layer.apply_gradients(0.05)
    end
    out = layer.forward(input)
    out[0, 0].should be_close(1.0, 0.1)
    out[1, 1].should be_close(1.0, 0.1)
  end
end

describe "Network with TransformerLayer" do
  it "can overfit a small sequence" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    net = SHAInet::Network.new
    net.add_layer(:input, 2, :memory, SHAInet.none)
    net.add_layer(:transformer, 2)
    net.add_layer(:output, 2, :memory, SHAInet.none)
    training = [[[[1.0, 0.0]], [1.0, 1.0]]]
    net.learning_rate = 0.1
    net.train(data: training, training_type: :sgdm,
      epochs: 2000, mini_batch_size: 1, log_each: 2000)
    out = net.run([[1.0, 0.0]]).last
    out[0].should be_close(1.0, 0.1)
    out[1].should be_close(1.0, 0.1)
  end
end
