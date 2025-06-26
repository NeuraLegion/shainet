require "./spec_helper"

describe SHAInet::LayerNorm do
  it "normalizes rows" do
    ln = SHAInet::LayerNorm.new(2)
    x = SHAInet::SimpleMatrix.from_a([[1.0, 3.0], [2.0, 0.0]])
    out = ln.forward(x)
    out.rows.times do |i|
      mean = 0.0
      var = 0.0
      out.cols.times { |j| mean += out[i, j] }
      mean /= out.cols
      out.cols.times do |j|
        diff = out[i, j] - mean
        var += diff*diff
      end
      var /= out.cols
      mean.should be_close(0.0, 1e-6)
      var.should be_close(1.0, 1e-4)
    end
  end
end

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

  it "respects an attention mask" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    attn = SHAInet::MultiHeadAttention.new(2, 1)
    input = SHAInet::SimpleMatrix.from_a([[1.0, 0.0], [0.0, 1.0]])
    mask = SHAInet::SimpleMatrix.from_a([[0.0, -1e9], [-1e9, 0.0]])
    out = attn.forward(input, mask)
    expected = (input * attn.w_v) * attn.w_o
    out.rows.times do |i|
      out.cols.times do |j|
        out[i, j].should be_close(expected[i, j], 1e-6)
      end
    end
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
    net.fully_connect
    training = [[[[1.0, 0.0]], [1.0, 1.0]]]
    net.learning_rate = 0.005
    net.train(data: training, training_type: :sgdm,
      epochs: 20_000, mini_batch_size: 1, log_each: 2000)
    out = net.run([[1.0, 0.0]]).last
    out[0].should be > 0.5
    out[1].should be > 0.5
  end

  it "works with embeddings and positional encoding" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    net = SHAInet::Network.new
    net.add_layer(:input, 1, :memory, SHAInet.none)
    net.add_layer(:embedding, 2, :memory, SHAInet.none, vocab_size: 3)
    net.add_layer(:transformer, 2)
    net.add_layer(:output, 2, :memory, SHAInet.none)
    net.fully_connect

    pe = SHAInet::PositionalEncoding.sinusoidal(2, 2)
    net.transformer_layers.each { |l| l.positional_encoding = pe }

    out = net.run([[1.0], [2.0]]).last
    out.size.should eq(2)
  end
end
