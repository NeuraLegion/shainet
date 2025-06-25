require "./spec_helper"

private def f(x : Float64)
  x * x + 3.0
end

describe SHAInet::Autograd::Tensor do
  it "matches numerical gradient" do
    x = SHAInet::Autograd::Tensor.new(2.0)
    y = x * x + 3
    y.backward
    autograd_grad = x.grad

    h = 1e-6
    numeric = (f(x.data + h) - f(x.data - h)) / (2*h)
    autograd_grad.should be_close(numeric, 1e-6)
  end
end
