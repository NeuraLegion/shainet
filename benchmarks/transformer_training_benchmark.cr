require "../src/shainet"

module SHAInet::CUDA
  def self.reset_check
    @@checked = false
  end
end

text = "hello world hello world"

tokenizer = SHAInet::BPETokenizer.new
vocab_size = 30
tokenizer.train(text, vocab_size)
ids = tokenizer.encode(text)

token_count = tokenizer.vocab.size

def build_training(ids, token_count)
  training = [] of Array(Array(Array(Int32)) | Array(Float64))
  (0...ids.size - 1).each do |i|
    seq = [[ids[i]]]
    target = Array(Float64).new(token_count, 0.0)
    target[ids[i + 1]] = 1.0
    training << [seq, target]
  end
  training
end

training = build_training(ids, token_count)

def build_net(token_count)
  net = SHAInet::Network.new
  net.add_layer(:input, 1, :memory, SHAInet.none)
  net.add_layer(:embedding, 8, :memory, SHAInet.none, vocab_size: token_count)
  net.add_layer(:transformer, 8)
  net.add_layer(:output, token_count, :memory, SHAInet.sigmoid)
  net.fully_connect
  net.learning_rate = 0.001
  net
end

# Collect matrices to measure GPU memory usage

def collect_mats(net : SHAInet::Network)
  mats = [] of SHAInet::SimpleMatrix
  layers = net.input_layers + net.hidden_layers + net.output_layers
  layers.each do |l|
    mats << l.weights
    mats << l.biases
    if l.is_a?(SHAInet::EmbeddingLayer)
      mats << l.as(SHAInet::EmbeddingLayer).embeddings
    elsif l.is_a?(SHAInet::TransformerLayer)
      blk = l.as(SHAInet::TransformerLayer)
      mats << blk.mha.w_q
      mats << blk.mha.w_k
      mats << blk.mha.w_v
      mats << blk.mha.w_o
      mats << blk.ffn.w1
      mats << blk.ffn.b1
      mats << blk.ffn.w2
      mats << blk.ffn.b2
      mats << blk.norm1.gamma
      mats << blk.norm1.beta
      mats << blk.norm2.gamma
      mats << blk.norm2.beta
    end
  end
  mats
end

# Benchmark helper

def run_bench(use_gpu : Bool, token_count : Int32, training)
  if use_gpu
    ENV.delete("SHAINET_DISABLE_CUDA")
  else
    ENV["SHAINET_DISABLE_CUDA"] = "1"
  end
  SHAInet::CUDA.reset_check
  label = use_gpu ? "GPU" : "CPU"
  net = build_net(token_count)
  puts "#{label} benchmark:"
  epochs = 3
  epochs.times do |i|
    start = Time.monotonic
    net.train(data: training, training_type: :adamw, cost_function: :c_ent, epochs: 1, mini_batch_size: 1, log_each: training.size)
    span = Time.monotonic - start
    mem = SHAInet::GPUMemory.estimate_gpu_memory_usage(collect_mats(net))
    puts "  Epoch #{i + 1}: #{span.total_milliseconds}ms, memory #{mem / 1024 / 1024}MB"
  end
end

# CPU first
run_bench(false, token_count, training)

if SHAInet::CUDA.available?
  run_bench(true, token_count, training)
else
  puts "CUDA not available, skipping GPU run"
end
