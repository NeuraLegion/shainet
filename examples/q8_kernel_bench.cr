require "../src/shainet"

# Isolate Q8 kernel throughput from the inference-path overhead.
unless SHAInet::CUDA.fully_available?
  STDERR.puts "no cuda"; exit 1
end

def bench(name, k, n, iters)
  w = SHAInet::SimpleMatrix.new(k, n)
  k.times { |r| n.times { |c| w[r, c] = (Random.rand * 2 - 1) } }
  qw = SHAInet::QuantizedCudaMatrix.from_simple(w)
  x = SHAInet::SimpleMatrix.new(1, k)
  k.times { |c| x[0, c] = (Random.rand * 2 - 1) }
  x_gpu = x.to_cuda # upload once, keep on device

  # warmup
  3.times { qw.gemv(x_gpu) }

  t = Time.instant
  iters.times { qw.gemv(x_gpu) }
  el = (Time.instant - t).total_seconds
  printf("%-10s K=%-5d N=%-6d  %6.3f ms/call  (%d iters)\n", name, k, n, el / iters * 1000, iters)
end

# The 8 GEMVs that make up one decode token for LLaMA 3.2 1B:
bench("q_proj", 2048, 2048, 500)
bench("k_proj", 2048, 512, 500)
bench("v_proj", 2048, 512, 500)
bench("o_proj", 2048, 2048, 500)
bench("gate", 2048, 8192, 500)
bench("up", 2048, 8192, 500)
bench("down", 8192, 2048, 500)
bench("lm_head", 2048, 128256, 200)
