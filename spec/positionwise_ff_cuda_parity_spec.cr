require "./spec_helper"

describe "PositionWiseFF GPU parity" do
  it "matches CPU and GPU bias gradients" do
    handle_ok = begin
      h = SHAInet::CUDA.create_handle
      SHAInet::CUDA.destroy_handle(h)
      true
    rescue
      false
    end
    pending! "CUDA not available" unless handle_ok
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    cpu_ff = SHAInet::PositionWiseFF.new(4, 6)
    x_cpu = SHAInet::SimpleMatrix.from_a([[1.0, 0.5, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    dout_cpu = SHAInet::SimpleMatrix.ones(2, 4)
    cpu_ff.forward(x_cpu)
    cpu_ff.backward(dout_cpu)
    gb1_cpu = cpu_ff.g_b1.clone
    gb2_cpu = cpu_ff.g_b2.clone

    ENV.delete("SHAINET_DISABLE_CUDA")
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    gpu_ff = SHAInet::PositionWiseFF.new(4, 6)
    x_gpu = SHAInet::CudaMatrix.from_a(x_cpu.to_a)
    dout_gpu = SHAInet::CudaMatrix.ones(2, 4)
    gpu_ff.forward(x_gpu)
    gpu_ff.backward(dout_gpu)
    gb1_gpu = gpu_ff.g_b1.clone
    gb2_gpu = gpu_ff.g_b2.clone

    gb1_gpu.rows.times do |i|
      gb1_gpu.cols.times do |j|
        gb1_gpu[i, j].should be_close(gb1_cpu[i, j], 1e-6)
      end
    end

    gb2_gpu.rows.times do |i|
      gb2_gpu.cols.times do |j|
        gb2_gpu[i, j].should be_close(gb2_cpu[i, j], 1e-6)
      end
    end
  end
end
