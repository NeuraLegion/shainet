require "./spec_helper"

describe SHAInet do
  it "computes cross entropy cost derivative for expected 1" do
    expected = 1.0
    actual = 0.8
    eps = 1e-6
    forward = SHAInet._cross_entropy_cost(expected, actual + eps)
    backward = SHAInet._cross_entropy_cost(expected, actual - eps)
    numeric = (forward - backward) / (2 * eps)
    formula = SHAInet._cross_entropy_cost_derivative(expected, actual)
    formula.should be_close(numeric, 1e-5)
  end

  it "computes cross entropy cost derivative for expected 0" do
    expected = 0.0
    actual = 0.2
    eps = 1e-6
    forward = SHAInet._cross_entropy_cost(expected, actual + eps)
    backward = SHAInet._cross_entropy_cost(expected, actual - eps)
    numeric = (forward - backward) / (2 * eps)
    formula = SHAInet._cross_entropy_cost_derivative(expected, actual)
    formula.should be_close(numeric, 1e-5)
  end
end
