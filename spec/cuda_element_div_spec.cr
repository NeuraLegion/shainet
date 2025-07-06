require "./spec_helper"

describe "CUDA element-wise division" do
  it "matches CPU division and handles divide by zero" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    a = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a([[4.0, 8.0], [3.0, 6.0]]))
    b = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a([[2.0, 0.0], [3.0, 2.0]]))

    result = a.as(SHAInet::CudaMatrix) / b.as(SHAInet::CudaMatrix)
    result.sync_from_device!

    result[0, 0].should eq(2.0)
    result[0, 1].should eq(0.0)
    result[1, 0].should eq(1.0)
    result[1, 1].should eq(3.0)
  end
end
