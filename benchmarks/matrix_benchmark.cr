require "../src/shainet"

rows = 512
cols = 512
puts "CUDA available: #{SHAInet::CUDA.available?}"
puts "cuDNN available: #{SHAInet::CUDA.cudnn_available?}"

cpu_a = SHAInet::SimpleMatrix.new(rows, cols).random_fill!
cpu_b = SHAInet::SimpleMatrix.new(cols, cols).random_fill!
start = Time.monotonic
cpu_a * cpu_b
cpu_mul = Time.monotonic - start

gpu_a = SHAInet::CudaMatrix.new(rows, cols).random_fill!
gpu_b = SHAInet::CudaMatrix.new(cols, cols).random_fill!
start = Time.monotonic
gpu_a * gpu_b
gpu_mul = Time.monotonic - start

m_cpu = SHAInet::SimpleMatrix.new(rows, cols).random_fill!(-1.0, 1.0)
start = Time.monotonic
m_cpu.relu!
cpu_relu = Time.monotonic - start

m_gpu = SHAInet::CudaMatrix.new(rows, cols).random_fill!(-1.0, 1.0)
start = Time.monotonic
m_gpu.relu!
gpu_relu = Time.monotonic - start

puts "Matrix multiply CPU: #{cpu_mul.total_milliseconds}ms"
puts "Matrix multiply GPU: #{gpu_mul.total_milliseconds}ms"
puts "ReLU CPU: #{cpu_relu.total_milliseconds}ms"
puts "ReLU GPU: #{gpu_relu.total_milliseconds}ms"
