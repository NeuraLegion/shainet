require "./spec_helper"

describe SHAInet::RNNDropout do
  it "drops approximately the given percentage of values" do
    arr = Array(Float64).new(100, 1.0)
    runs = 1000
    total_ratio = 0.0
    runs.times do
      out = SHAInet::RNNDropout.apply(arr, 30)
      dropped = out.count(0.0)
      total_ratio += dropped.to_f / out.size
    end
    average = total_ratio / runs
    (average).should be_close(0.30, 0.05)
  end
end
