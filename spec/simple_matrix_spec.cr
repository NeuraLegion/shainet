require "./spec_helper"

describe SHAInet::SimpleMatrix do
  it "supports basic operations" do
    a = SHAInet::SimpleMatrix.new(2, 2)
    a[0,0] = 1.0; a[0,1] = 2.0
    a[1,0] = 3.0; a[1,1] = 4.0

    b = SHAInet::SimpleMatrix.new(2,2)
    b[0,0] = 1.0; b[1,1] = 1.0

    sum = a + b
    sum[1,1].should eq(5.0)

    prod = a * b
    prod[0,0].should eq(1.0)
    prod[1,1].should eq(4.0)

    t = a.transpose
    t[0,1].should eq(3.0)
  end
end

