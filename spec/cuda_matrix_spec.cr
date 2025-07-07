require "./spec_helper"

describe SHAInet::CudaMatrix do
  it "mirrors SimpleMatrix operations" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    a = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a([[1, 2], [3, 4]]))
    b = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a([[1, 0], [0, 1]]))

    sum = a.as(SHAInet::CudaMatrix) + b.as(SHAInet::CudaMatrix)
    sum[1, 1].should eq(5.0)

    prod = a.as(SHAInet::CudaMatrix) * b.as(SHAInet::CudaMatrix)
    prod[0, 0].should eq(1.0)
    prod[1, 1].should eq(4.0)

    t = a.transpose
    t[0, 1].should eq(3.0)
  end

  it "performs relu and add_bias on GPU when available" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    matrix = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a([[-1, 2], [-3, 4]]))
    bias = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a([[1, 1]]))

    matrix.as(SHAInet::CudaMatrix).relu!
    matrix.as(SHAInet::CudaMatrix).add_bias!(bias.as(SHAInet::CudaMatrix))

    if SHAInet::CUDA.fully_available?
      matrix.as(SHAInet::CudaMatrix).device_ptr.should_not be_nil
    end

    matrix[0, 0].should eq(0.0 + 1.0)
    matrix[0, 1].should eq(2.0 + 1.0)
    matrix[1, 0].should eq(0.0 + 1.0)
    matrix[1, 1].should eq(4.0 + 1.0)
  end
end
