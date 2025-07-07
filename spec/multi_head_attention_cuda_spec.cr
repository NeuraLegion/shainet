require "./spec_helper"

describe "MultiHeadAttention GPU parity" do
  it "matches CPU implementation" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    cpu_attn = SHAInet::MultiHeadAttention.new(2, 1)
    input = SHAInet::SimpleMatrix.from_a([[1.0, 0.0], [0.0, 1.0]])
    cpu_out = cpu_attn.forward(input)
    ENV.delete("SHAINET_DISABLE_CUDA")
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    gpu_attn = SHAInet::MultiHeadAttention.new(2, 1)
    gpu_in = SHAInet::GPUMemory.to_gpu(input)
    gpu_out = gpu_attn.forward(gpu_in)

    # Ensure GPU results are synced to host
    if gpu_out.is_a?(SHAInet::CudaMatrix)
      gpu_out.sync_from_device!
    end

    cpu_out.rows.times do |i|
      cpu_out.cols.times do |j|
        gpu_out[i, j].should be_close(cpu_out[i, j], 1e-6)
      end
    end
  end
end
