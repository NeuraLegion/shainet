require "./spec_helper"

describe SHAInet::CudaMatrix do
  it "mirrors SimpleMatrix operations" do
    a = SHAInet::CudaMatrix.from_a([[1, 2], [3, 4]])
    b = SHAInet::CudaMatrix.from_a([[1, 0], [0, 1]])

    sum = a + b
    sum[1, 1].should eq(5.0)

    prod = a * b
    prod[0, 0].should eq(1.0)
    prod[1, 1].should eq(4.0)

    t = a.transpose
    t[0, 1].should eq(3.0)
  end

  it "performs relu and add_bias on GPU when available" do
    matrix = SHAInet::CudaMatrix.from_a([[-1, 2], [-3, 4]])
    bias = SHAInet::CudaMatrix.from_a([[1, 1]])

    matrix.relu!
    matrix.add_bias!(bias)

    if SHAInet::CUDA.available?
      matrix.device_ptr.should_not be_nil
    end

    matrix[0, 0].should eq(0.0 + 1.0)
    matrix[0, 1].should eq(2.0 + 1.0)
    matrix[1, 0].should eq(0.0 + 1.0)
    matrix[1, 1].should eq(4.0 + 1.0)
  end
end
