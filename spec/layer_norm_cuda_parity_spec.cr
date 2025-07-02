require "./spec_helper"

describe "LayerNorm GPU parity" do
  it "matches CPU and GPU forward/backward" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    rows = 3
    cols = 4

    data = Array.new(rows) { Array.new(cols) { rand } }
    dout_data = Array.new(rows) { Array.new(cols) { rand } }

    # CPU-only version
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    cpu_ln = SHAInet::LayerNorm.new(cols)
    x_cpu = SHAInet::SimpleMatrix.from_a(data)
    dout_cpu = SHAInet::SimpleMatrix.from_a(dout_data)
    out_cpu = cpu_ln.forward(x_cpu)
    dx_cpu = cpu_ln.backward(dout_cpu)
    g_gamma_cpu = cpu_ln.g_gamma.clone
    g_beta_cpu = cpu_ln.g_beta.clone
    ENV.delete("SHAINET_DISABLE_CUDA")

    # GPU version with same parameters
    gpu_ln = SHAInet::LayerNorm.new(cols)
    # Copy the exact same gamma and beta values from CPU version
    cols.times do |j|
      gpu_ln.gamma[0, j] = cpu_ln.gamma[0, j]
      gpu_ln.beta[0, j] = cpu_ln.beta[0, j]
    end

    x_gpu = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a(data))
    dout_gpu = SHAInet::GPUMemory.to_gpu(SHAInet::SimpleMatrix.from_a(dout_data))

    out_gpu = gpu_ln.forward(x_gpu)

    # Sync GPU results if needed
    SHAInet::GPUMemory.batch_sync_from_device([out_gpu]) if out_gpu.is_a?(SHAInet::CudaMatrix)

    dx_gpu = gpu_ln.backward(dout_gpu)
    SHAInet::GPUMemory.batch_sync_from_device([dx_gpu]) if dx_gpu.is_a?(SHAInet::CudaMatrix)

    rows.times do |i|
      cols.times do |j|
        out_gpu[i, j].should be_close(out_cpu[i, j], 1e-6)
        dx_gpu[i, j].should be_close(dx_cpu[i, j], 1e-6)
      end
    end

    cols.times do |j|
      gpu_ln.g_gamma[0, j].should be_close(g_gamma_cpu[0, j], 1e-6)
      gpu_ln.g_beta[0, j].should be_close(g_beta_cpu[0, j], 1e-6)
    end
  end
end
