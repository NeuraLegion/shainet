require "../src/shainet"

# Minimal Transformer language model example
# -----------------------------------------
# 1. Tokenize some text with BPETokenizer
# 2. Build a network (Embedding -> Transformer -> Output)
# 3. Train using cross-entropy loss
# 4. Predict the next token

text = "hello world hello world"

# Train tokenizer and encode text
tokenizer = SHAInet::BPETokenizer.new
vocab_size = 30
tokenizer.train(text, vocab_size)
ids = tokenizer.encode(text)

token_count = tokenizer.vocab.size

net = SHAInet::Network.new
net.add_layer(:input, 1, :memory, SHAInet.none)
net.add_layer(:embedding, 8, :memory, SHAInet.none)
net.add_layer(:transformer, 8)
net.add_layer(:output, token_count, :memory, SHAInet.sigmoid)
net.fully_connect
net.warmup_steps = 10
net.weight_decay = 0.01
net.accumulation_steps = 2
net.mixed_precision = true

# Helper to create one-hot vectors
one_hot = ->(id : Int32, size : Int32) do
  arr = Array(Float64).new(size, 0.0)
  arr[id] = 1.0
  arr
end

# Each token predicts the next token
training = [] of Tuple(Array(Array(Float64)), Array(Float64))
(0...ids.size - 1).each do |i|
  input = [[ids[i].to_f64]]
  expected = one_hot.call(ids[i + 1], token_count)
  training << {input, expected}
end

# Convert tuples to arrays for training
train_data = training.map { |seq, target| [seq, target] }

net.learning_rate = 0.001
net.train(data: train_data,
  training_type: :adamw,
  cost_function: :c_ent,
  epochs: 200,
  mini_batch_size: 1,
  log_each: 50)

# Predict the token following "hello"
hello_id = tokenizer.encode("hello").first
output = net.run([[hello_id.to_f64]]).last
pred_id = output.index(output.max) || 0
puts "Prediction for 'hello' -> #{tokenizer.decode([pred_id])}"
