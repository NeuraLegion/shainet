require "../src/shainet"
# BabyLM challenge example (GPU-optimized)
# ------------------------
# This example has been optimized for better GPU utilization.
#
# To enable full GPU acceleration:
# 1. Build CUDA kernels: ./build_cuda_kernels.sh
# 2. Set library path: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
# 3. Monitor GPU usage: nvidia-smi
#
# 1. Download the BabyLM training set from the following URL:
#    https://osf.io/ryjfm/files/osfstorage/6819fdbfbecda878d4c61566 (train_100M.zip)
#    Extract `train.txt` somewhere locally (for this example we expect it under
#    `data/train.txt`).
# 2. Train a tokenizer on the dataset.
# 3. Build a Transformer based language model with positional encoding.
# 4. Train it using cross-entropy loss.
# 5. Predict the next token for a sample input.

# Path to the unzipped training text
path = "/home/unshadow/Downloads/train_100M/childes.train"
puts "Reading dataset from #{path}..."
text = File.read(path)
puts "Dataset loaded, size: #{text.size} characters."

puts "Using the GPU? #{SHAInet::CUDA.available? ? "Yes" : "No"}"
puts "Kernels available? #{SHAInet::CUDA.kernels_available? ? "Yes" : "No"}"
puts "Training the tokenizer on the dataset..."
# Train tokenizer and encode text
vocab_size = 10_000
tokenizer = SHAInet::BPETokenizer.new
tokenizer.train(text, vocab_size)
ids = tokenizer.encode(text)

puts "Tokenizer trained with #{tokenizer.vocab.size} tokens."

puts "Building the network..."
# Build the network
d_model = 256
seq_len = 16
token_count = tokenizer.vocab.size
net = SHAInet::Network.new
net.add_layer(:input, 1, :memory, SHAInet.none)
net.add_layer(:embedding, d_model, :memory, SHAInet.none, vocab_size: token_count)
4.times { net.add_layer(:transformer, d_model) }
# Use a sigmoid output so cross-entropy can be applied per token
net.add_layer(:output, token_count, :memory, SHAInet.identity)
net.fully_connect

puts "Network built"
# Positional encoding should only be applied to the first transformer layer
pos_enc = SHAInet::PositionalEncoding.sinusoidal(seq_len, d_model)
net.transformer_layers.first.positional_encoding = pos_enc

# Build training/validation splits and write pairs to disk for streaming
def write_pairs(path, ids, seq_len)
  File.open(path, "w") do |f|
    (0...(ids.size - seq_len)).each do |i|
      seq = ids[i, seq_len].map { |id| [id] }
      pair = [seq, [ids[i + seq_len]]]
      f.puts pair.to_json
    end
  end
end

split = ids.size * 9 // 10
train_ids = ids[0, split]
val_ids = ids[split - seq_len, ids.size - (split - seq_len)]

train_file = "train_pairs.jsonl"
val_file = "val_pairs.jsonl"

write_pairs(train_file, train_ids, seq_len)
write_pairs(val_file, val_ids, seq_len)

train_data = SHAInet::StreamingData.new(train_file, shuffle: true, gpu_batches: true)
val_data = SHAInet::StreamingData.new(val_file, gpu_batches: true)

epochs = 10
batch = 32
net.learning_rate = 0.001

puts "Training the network for #{epochs} epochs with batch size #{batch}..."
epochs.times do |epoch|
  # Use StreamingData without mini_batch_size to avoid CPU bottlenecks
  net.train(data: train_data,
    training_type: :adam,
    cost_function: :c_ent_sm,
    epochs: 1,
    log_each: 100)

  # Optimized validation with GPU batch processing
  val_loss = 0.0
  count = 0
  
  # Process validation in larger batches for better GPU utilization
  val_batch_size = 64
  while (val_batch = val_data.next_batch(val_batch_size)).size > 0
    total_batch_loss = 0.0
    
    val_batch.each do |sample|
      seq = sample[0].as(Array(Array(Float64)))
      tgt = sample[1].as(Array(Float64)).first.to_i
      output_vec = net.run(seq).last

      # Use native softmax - it's already optimized
      probs = SHAInet.softmax(output_vec)
      total_batch_loss += -Math.log(probs[tgt].clamp(1e-9, 1.0))
      count += 1
    end
    
    val_loss += total_batch_loss
  end
  val_loss /= count.to_f if count > 0
  val_data.rewind
  puts "Epoch #{epoch + 1} validation loss: #{val_loss.round(4)}"
end

# Predict the token following the first token in the dataset
first_id = ids.first
output = net.run([[first_id]]).last
pred_id = output.index(output.max) || 0
puts "Prediction -> #{tokenizer.decode([pred_id])}"
