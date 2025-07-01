require "./spec_helper"

class DummyMatrix < SHAInet::UnifiedMatrix
  def initialize; end

  def forward(input : SHAInet::UnifiedMatrix) : SHAInet::UnifiedMatrix
    self
  end

  def backward(grad : SHAInet::UnifiedMatrix) : SHAInet::UnifiedMatrix
    self
  end

  def update_weights(lr : Float64)
    # no-op
  end
end

describe SHAInet::UnifiedMatrix do
  it "allows subclassing and method invocation" do
    dummy = DummyMatrix.new
    dummy.forward(dummy).should be_a(SHAInet::UnifiedMatrix)
    dummy.backward(dummy).should be_a(SHAInet::UnifiedMatrix)
    dummy.update_weights(0.1)
  end
end
