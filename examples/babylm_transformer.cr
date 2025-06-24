require "../src/shainet"

# BabyLM challenge example
# ------------------------
# 1. Download the BabyLM training set from the following URL:
#    https://osf.io/ryjfm/files/osfstorage/6819fdbfbecda878d4c61566 (train_100M.zip)
#    Extract `train.txt` somewhere locally (for this example we expect it under
#    `data/train.txt`).
# 2. Train a tokenizer on the dataset.
# 3. Build a Transformer based language model with positional encoding.
# 4. Train it using cross-entropy loss.
# 5. Predict the next token for a sample input.

# Path to the unzipped training text
path = "data/train.txt"
text = File.read(path)

# Train tokenizer and encode text
vocab_size = 30_000
tokenizer = SHAInet::BPETokenizer.new
tokenizer.train(text, vocab_size)
ids = tokenizer.encode(text)

# Build the network
d_model = 256
seq_len = 16
token_count = tokenizer.vocab.size
net = SHAInet::Network.new
net.add_layer(:input, 1, :memory, SHAInet.none)
net.add_layer(:embedding, d_model, :memory, SHAInet.none)
4.times { net.add_layer(:transformer, d_model) }
net.add_layer(:output, token_count, :memory, SHAInet.softmax)
net.fully_connect

# Positional encoding shared across layers
pos_enc = SHAInet::PositionalEncoding.sinusoidal(seq_len, d_model)
net.transformer_layers.each { |l| l.positional_encoding = pos_enc }

# Helper for one-hot vectors
def one_hot(id, size)
  arr = Array(Float64).new(size, 0.0)
  arr[id] = 1.0
  arr
end

# Build training/validation splits and helper to create pairs
def build_pairs(ids, seq_len, vocab_size)
  pairs = [] of Tuple(Array(Array(Int32)), Array(Float64))
  (0...(ids.size - seq_len)).each do |i|
    seq = ids[i, seq_len].map { |id| [id] }
    target = Array(Float64).new(vocab_size, 0.0)
    target[ids[i + seq_len]] = 1.0
    pairs << {seq, target}
  end
  pairs
end

split = ids.size * 9 // 10
train_ids = ids[0, split]
val_ids = ids[split - seq_len, ids.size - (split - seq_len)]

training = build_pairs(train_ids, seq_len, token_count)
validation = build_pairs(val_ids, seq_len, token_count)

epochs = 10
batch = 32
net.learning_rate = 0.001

epochs.times do |epoch|
  training.shuffle!
  net.train(data: training,
    training_type: :adam,
    cost_function: :c_ent,
    epochs: 1,
    mini_batch_size: batch,
    log_each: 1000)

  val_loss = 0.0
  validation.each do |seq, expected|
    output_vec = net.run(seq).last
    tgt = expected.index(1.0) || 0
    val_loss += -Math.log(output_vec[tgt].clamp(1e-9, 1.0))
  end
  val_loss /= validation.size.to_f
  puts "Epoch #{epoch + 1} validation loss: #{val_loss}"
end

# Predict the token following the first token in the dataset
first_id = ids.first
output = net.run([[first_id]]).last
pred_id = output.index(output.max) || 0
puts "Prediction -> #{tokenizer.decode([pred_id])}"
