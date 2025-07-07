require "./spec_helper"

describe SHAInet::Autograd::Tensor do
  it "computes gradients for simple operations" do
    a = SHAInet::Autograd::Tensor.new(2.0)
    b = SHAInet::Autograd::Tensor.new(3.0)
    c = a * b + a
    c.backward

    a.grad.should eq(3.0 + 1.0)
    b.grad.should eq(2.0)
    c.grad.should eq(1.0)
  end

  it "computes gradients for matrix multiply" do
    x = SHAInet::Autograd::Tensor.new(2.0)
    y = SHAInet::Autograd::Tensor.new(4.0)
    z = x.matmul(y)
    z.backward

    x.grad.should eq(4.0)
    y.grad.should eq(2.0)
  end
end
