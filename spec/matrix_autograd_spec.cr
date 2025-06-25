require "./spec_helper"

describe SHAInet::SimpleMatrix do
  it "propagates gradients through matrix operations" do
    a = SHAInet::SimpleMatrix.tensor(1, 2)
    a[0, 0] = SHAInet::Autograd::Tensor.new(2.0)
    a[0, 1] = SHAInet::Autograd::Tensor.new(3.0)

    b = SHAInet::SimpleMatrix.tensor(2, 1)
    b[0, 0] = SHAInet::Autograd::Tensor.new(4.0)
    b[1, 0] = SHAInet::Autograd::Tensor.new(5.0)

    out = a * b
    out[0, 0].as(SHAInet::Autograd::Tensor).backward

    a[0, 0].as(SHAInet::Autograd::Tensor).grad.should eq(4.0)
    a[0, 1].as(SHAInet::Autograd::Tensor).grad.should eq(5.0)
    b[0, 0].as(SHAInet::Autograd::Tensor).grad.should eq(2.0)
    b[1, 0].as(SHAInet::Autograd::Tensor).grad.should eq(3.0)
  end
end
