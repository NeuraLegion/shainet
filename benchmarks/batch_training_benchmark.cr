require "../src/shainet"

# Simple benchmark comparing training with individual sequences vs pre-batched sequences

SAMPLES = 50
SEQ_LEN =  2
DIM     =  2

# Generate random data
random_seq = -> {
  Array.new(SEQ_LEN) { Array.new(DIM) { rand } }
}
random_out = -> { Array.new(DIM) { rand } }

data = Array.new(SAMPLES) { [random_seq.call, random_out.call] }

# Build a tiny network
net = SHAInet::Network.new
net.add_layer(:input, DIM, :memory, SHAInet.none)
net.add_layer(:transformer, DIM)
net.add_layer(:output, DIM, :memory, SHAInet.none)
net.fully_connect
net.learning_rate = 0.001

# Train without batching
start = Time.monotonic
net.train(data: data, training_type: :sgdm, epochs: 2, mini_batch_size: 1)
no_batch_time = Time.monotonic - start

# Train with batch wrapper (each batch is the full data set)
start = Time.monotonic
net.train(data: [data], training_type: :sgdm, epochs: 2, mini_batch_size: 1)
with_batch_time = Time.monotonic - start

puts "No batching: #{no_batch_time.total_milliseconds}ms"
puts "With batching: #{with_batch_time.total_milliseconds}ms"
