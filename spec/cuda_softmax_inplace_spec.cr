require "./spec_helper"

describe "CudaMatrix#softmax_rows!" do
  it "matches CPU softmax" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    cpu = SHAInet::SimpleMatrix.from_a([[0.5, 1.5, 2.5], [3.0, 0.0, -1.0]])
    expected = SHAInet.softmax_rows(cpu)

    gpu = SHAInet::GPUMemory.to_gpu(cpu).as(SHAInet::CudaMatrix)
    gpu.softmax_rows!
    gpu.sync_from_device!

    expected.rows.times do |i|
      expected.cols.times do |j|
        gpu[i, j].should be_close(expected[i, j], 1e-6)
      end
    end
  end
end
