require "../src/shainet"

ROWS  =  256
COLS  =  256
COUNT = 1000

puts "CUDA available: #{SHAInet::CUDA.available?}"

def allocate(count)
  count.times do
    m = SHAInet::CudaMatrix.new(ROWS, COLS)
    m.finalize
  end
end

# Without pooling
SHAInet::GPUMemory.pool_limit = 0
SHAInet::GPUMemory.cleanup
start = Time.monotonic
allocate(COUNT)
no_pool = Time.monotonic - start

# With pooling
SHAInet::GPUMemory.pool_limit = COUNT
SHAInet::GPUMemory.preallocate!(ROWS, COLS, COUNT)
start = Time.monotonic
allocate(COUNT)
with_pool = Time.monotonic - start

puts "Allocate #{COUNT} matrices without pool: #{no_pool.total_milliseconds}ms"
puts "Allocate #{COUNT} matrices with pool: #{with_pool.total_milliseconds}ms"
