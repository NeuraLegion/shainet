require "./spec_helper"

describe "CUDA element_log" do
  it "matches CPU log" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.kernels_available?

    cpu = SHAInet::SimpleMatrix.from_a([[0.5, 1.0], [2.0, 4.0]])
    gpu = SHAInet::GPUMemory.to_gpu(cpu)

    gpu_out = SHAInet::CudaMatrix.new(cpu.rows, cpu.cols)
    SHAInet::CUDNN.element_log!(gpu_out, gpu.as(SHAInet::CudaMatrix))
    gpu_out.sync_from_device!

    cpu_out = SHAInet::SimpleMatrix.new(cpu.rows, cpu.cols)
    cpu.rows.times do |i|
      cpu.cols.times do |j|
        cpu_out[i, j] = Math.log(cpu[i, j])
      end
    end

    cpu.rows.times do |i|
      cpu.cols.times do |j|
        gpu_out[i, j].should be_close(cpu_out[i, j], 1e-6)
      end
    end
  end
end
