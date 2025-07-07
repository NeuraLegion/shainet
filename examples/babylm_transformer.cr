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
vocab_size = 10000 # Much smaller vocab for faster training
tokenizer = SHAInet::BPETokenizer.new
tokenizer.train(text, vocab_size) # Much smaller dataset
ids = tokenizer.encode(text)

puts "Tokenizer trained with #{tokenizer.vocab.size} tokens."
puts "Dataset size: #{ids.size} tokens"

puts "Building the network..."
# Build the network with much smaller dimensions for fast debugging
d_model = 128
seq_len = 64
token_count = tokenizer.vocab.size
net = SHAInet::Network.new
net.add_layer(:input, 1, SHAInet.none)
net.add_layer(:embedding, d_model, SHAInet.none, vocab_size: token_count)
4.times { net.add_layer(:transformer, d_model) }
net.add_layer(:output, token_count, SHAInet.identity)
net.fully_connect

puts "Network built"
puts "Output layer size: #{token_count}"
puts "Embedding vocab size: #{token_count}"
# Positional encoding for the transformer layer
pos_enc = SHAInet::PositionalEncoding.sinusoidal(seq_len, d_model)
net.transformer_layers.first.positional_encoding = pos_enc

# Build training/validation splits and write pairs to disk for streaming

# Write pairs as much smaller JSONL: input is a sequence of token IDs, target is the next token ID (integer)
def write_pairs(path, ids, seq_len)
  File.open(path, "w") do |f|
    if ids.size <= seq_len
      puts "Warning: Dataset too small (#{ids.size} tokens) for sequence length #{seq_len}"
      return
    end

    max_id = ids.max
    puts "Max token ID in dataset: #{max_id}"

    (0...(ids.size - seq_len)).each do |i|
      seq = ids[i, seq_len]     # Array(Int32)
      target = ids[i + seq_len] # Int32
      # Write as {"input": [id, ...], "target": id}
      f.puts({"input" => seq, "target" => target}.to_json)
    end
  end
end

split = ids.size * 9 // 10
train_ids = ids[0, split]
val_ids = ids[split, ids.size - split]

train_file = "train_pairs.jsonl"
val_file = "val_pairs.jsonl"

write_pairs(train_file, train_ids, seq_len)
write_pairs(val_file, val_ids, seq_len)

puts "Training pairs written. Train size: #{train_ids.size}, Val size: #{val_ids.size}"
puts "Expected training sequences: #{train_ids.size - seq_len}"
puts "Expected validation sequences: #{val_ids.size - seq_len}"

# Data loader now expects {"input": [...], "target": ...} format.
train_data = SHAInet::StreamingData.new(train_file, shuffle: true, gpu_batches: true)
val_data = SHAInet::StreamingData.new(val_file, gpu_batches: true)

epochs = 100
batch = 1000 # Larger batch size for better GPU utilization
net.learning_rate = 0.001

puts "Training the network for #{epochs} epochs with batch size #{batch}..."
# Train for all epochs at once with proper logging
net.train(data: train_data,
  training_type: :adam,
  cost_function: :c_ent_sm,
  epochs: epochs,
  mini_batch_size: batch,
  log_each: 1) # Log every 5 epochs instead of every 100 samples

# Validation after training is complete
puts "Training complete. Running validation..."
val_loss = 0.0
count = 0

# Process validation in larger batches for better GPU utilization
val_batch_size = 64
while (val_batch = val_data.next_batch(val_batch_size)).size > 0
  total_batch_loss = 0.0

  val_batch.each do |sample|
    # sample is likely an Array: [input, target], but input can be various types
    input_raw = sample[0]
    input_ids = case input_raw
                when Array(Int32)
                  input_raw
                when Array(Array(Float64))
                  input_raw.map { |row| row[0].to_i }
                when Array(Float64)
                  input_raw.map(&.to_i)
                when SHAInet::CudaMatrix
                  input_raw.to_a.map { |row| row[0].to_i }
                when SHAInet::SimpleMatrix
                  input_raw.to_a.map { |row| row[0].to_i }
                else
                  raise "Unknown input type: #{input_raw.class}"
                end

    target_raw = sample[1]
    target_id = case target_raw
                when Int32
                  target_raw
                when Array(Float64)
                  target_raw.index(target_raw.max) || 0
                when Array(Array(Float64))
                  flat = target_raw.flatten
                  flat.index(flat.max) || 0
                when SHAInet::CudaMatrix
                  arr = target_raw.to_flat_array
                  arr.index(arr.max) || 0
                when SHAInet::SimpleMatrix
                  arr = target_raw.to_a.flatten
                  arr.index(arr.max) || 0
                else
                  raise "Unknown target type: #{target_raw.class}"
                end

    # Convert input_ids to [[id], [id], ...] for transformer
    seq = input_ids.map { |id| [id] }

    # Convert target_id to one-hot vector for loss calculation
    target = Array(Float64).new(token_count, 0.0)
    target[target_id] = 1.0

    output_vec = net.run(seq, return_matrix: true).as(SHAInet::CudaMatrix).to_a.last

    # Use native softmax - it's already optimized
    probs = SHAInet.softmax(output_vec)
    total_batch_loss += -Math.log(probs[target_id].clamp(1e-9, 1.0))
    count += 1
  end

  val_loss += total_batch_loss
end
val_loss /= count.to_f if count > 0
val_data.rewind
puts "Final validation loss: #{val_loss.round(4)}"

# Predict the token following a sequence from the dataset
test_seq = ids[0, seq_len].map { |id| [id] }
output = net.run(test_seq, return_matrix: true).as(SHAInet::CudaMatrix).to_a.last
pred_id = output.index(output.max) || 0
puts "Prediction -> #{tokenizer.decode([pred_id])}"
