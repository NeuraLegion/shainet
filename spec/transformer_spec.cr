require "./spec_helper"

describe SHAInet::MultiHeadAttention do
  it "trains to output constant values" do
    attn = SHAInet::MultiHeadAttention.new(2, 1)
    input = SHAInet::SimpleMatrix.from_a([[1.0, 0.0], [0.0, 1.0]])
    target = SHAInet::SimpleMatrix.ones(2, 2)
    500.times do
      out = attn.forward(input)
      diff = out - target
      attn.backward(diff, 0.05)
    end
    out = attn.forward(input)
    out[0, 0].should be_close(1.0, 0.1)
    out[1, 1].should be_close(1.0, 0.1)
  end
end

describe SHAInet::TransformerLayer do
  it "overfits a tiny sequence" do
    layer = SHAInet::TransformerLayer.new(2, 1, 4)
    input = SHAInet::SimpleMatrix.from_a([[1.0, 0.0], [0.0, 1.0]])
    target = SHAInet::SimpleMatrix.ones(2, 2)
    500.times do
      out = layer.forward(input)
      diff = out - target
      layer.backward(diff, 0.05)
    end
    out = layer.forward(input)
    out[0, 0].should be_close(1.0, 0.1)
    out[1, 1].should be_close(1.0, 0.1)
  end
end
