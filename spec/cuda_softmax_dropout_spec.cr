require "./spec_helper"

describe "CUDA softmax and dropout" do
  it "matches CPU softmax" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    cpu = SHAInet::SimpleMatrix.from_a([[1.0, 2.0], [3.0, 4.0]])
    gpu = SHAInet::GPUMemory.to_gpu(cpu)
    gpu_out = SHAInet.softmax_rows(gpu)
    SHAInet::GPUMemory.batch_sync_from_device([gpu_out])
    cpu_out = SHAInet.softmax_rows(cpu)
    cpu_out.rows.times do |i|
      cpu_out.cols.times do |j|
        gpu_out[i, j].should be_close(cpu_out[i, j], 1e-6)
      end
    end
  end

  it "drops approximately the given percentage" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    mat = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.ones(10, 10))
    runs = 200
    total_ratio = 0.0
    runs.times do |run_idx|
      out = SHAInet::TransformerDropout.apply(mat, 30)
      SHAInet::GPUMemory.batch_sync_from_device([out])

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
