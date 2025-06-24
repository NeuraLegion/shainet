require "./spec_helper"

describe SHAInet::TransformerDropout do
  it "drops approximately the given percentage of matrix entries" do
    mat = SHAInet::SimpleMatrix.ones(10, 10)
    runs = 1000
    total_ratio = 0.0
    runs.times do
      out = SHAInet::TransformerDropout.apply(mat, 30)
      dropped = 0
      mat.rows.times do |i|
        mat.cols.times do |j|
          dropped += 1 if out[i, j] == 0.0
        end
      end
      total_ratio += dropped.to_f / (mat.rows * mat.cols)
    end
    average = total_ratio / runs
    (average).should be_close(0.30, 0.05)
  end
end
