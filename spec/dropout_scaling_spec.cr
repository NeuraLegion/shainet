require "./spec_helper"
require "../src/shainet"

describe "Inverted dropout scaling" do
  it "preserves expected activation magnitude for SimpleMatrix" do
    # Large matrix of ones - after dropout, mean should still be ~1.0
    m = SHAInet::SimpleMatrix.new(100, 100, 1.0)
    m.dropout!(0.5)

    # Collect all values
    total = 0.0
    count = 0
    100.times do |i|
      100.times do |j|
        total += m[i, j]
        count += 1
      end
    end
    mean = total / count

    # With inverted scaling, kept values are 2.0 and dropped are 0.0
    # Expected mean ≈ 1.0 (within statistical tolerance for 10000 samples)
    mean.should be_close(1.0, 0.15)
  end

  it "zeros everything when prob is 1.0 for SimpleMatrix" do
    m = SHAInet::SimpleMatrix.new(10, 10, 5.0)
    m.dropout!(1.0)

    10.times do |i|
      10.times do |j|
        m[i, j].should eq(0.0)
      end
    end
  end

  it "preserves values when prob is 0.0 for SimpleMatrix" do
    m = SHAInet::SimpleMatrix.new(10, 10, 3.14)
    m.dropout!(0.0)

    10.times do |i|
      10.times do |j|
        m[i, j].should be_close(3.14, 1e-5)
      end
    end
  end
end
