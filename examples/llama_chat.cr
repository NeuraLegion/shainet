require "../src/shainet"
require "json"

# Simple LLaMA chat demo
# Usage:
#   crystal run examples/llama_chat.cr                          # auto-downloads SmolLM2-135M
#   crystal run examples/llama_chat.cr -- /path/to/model-dir   # use local model
#   crystal run examples/llama_chat.cr -- HuggingFaceTB/SmolLM2-135M  # download specific repo

model_dir = ARGV[0]? || "/tmp/smollm"
max_tokens = (ARGV[1]? || "50").to_i

# --- Minimal HF BPE Tokenizer ---
# --- Download model if needed ---
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"

def download_model(model_dir : String, repo : String)
  Dir.mkdir_p(model_dir)
  base_url = "https://huggingface.co/#{repo}/resolve/main"
  {"config.json", "model.safetensors", "tokenizer.json"}.each do |file|
    path = File.join(model_dir, file)
    next if File.exists?(path) && File.size(path) > 0
    STDERR.puts "  Downloading #{file}..."
    status = Process.run("curl", ["-sL", "#{base_url}/#{file}", "-o", path])
    raise "Failed to download #{file}" unless status.success?
  end
end

unless File.exists?(File.join(model_dir, "model.safetensors"))
  repo = ARGV[0]? || DEFAULT_MODEL
  STDERR.puts "Model not found at #{model_dir}, downloading #{repo}..."
  model_dir = File.join(Dir.tempdir, repo.gsub("/", "_"))
  download_model(model_dir, repo)
  STDERR.puts "Downloaded to #{model_dir}"
end

# --- Load model ---
STDERR.puts "Loading model from #{model_dir}..."
t = Time.monotonic
net = SHAInet::HFLoader.load_llama(model_dir)
STDERR.puts "Model loaded in #{(Time.monotonic - t).total_seconds.round(1)}s"
STDERR.puts "  Layers: #{net.transformer_layers.size}, d_model: #{net.transformer_layers.first.as(SHAInet::LlamaBlock).d_model}"

# --- Load tokenizer ---
tokenizer = SHAInet::BPETokenizer.from_hf(File.join(model_dir, "tokenizer.json"))
STDERR.puts "Tokenizer loaded (vocab: #{tokenizer.vocab.size})"
STDERR.puts ""

# --- Cached generation setup ---
emb = net.hidden_layers.find(&.is_a?(SHAInet::EmbeddingLayer)).as(SHAInet::EmbeddingLayer)
fn = net.final_norm.not_nil!
w = net.output_layers.first.weights.as(SHAInet::SimpleMatrix)
d_model = net.transformer_layers.first.as(SHAInet::LlamaBlock).d_model

# Move weights to GPU if available
if SHAInet::CUDA.fully_available?
  STDERR.puts "Moving to GPU..."
  net.transformer_layers.each { |l| l.as(SHAInet::LlamaBlock).to_gpu! }
  STDERR.puts "GPU ready!"
end

# --- Chat loop ---
loop do
  STDERR.print "You: "
  user_input = gets
  break if user_input.nil? || user_input.strip.empty?

  # Clear KV cache for new conversation
  net.transformer_layers.each { |l| l.as(SHAInet::LlamaBlock).clear_cache! }

  # Encode with BOS
  prompt_ids = [tokenizer.vocab["<|begin_of_text|>"]? || tokenizer.vocab["<s>"]? || 0] + tokenizer.encode(user_input.strip)
  STDERR.puts "(#{prompt_ids.size} tokens)"
  STDERR.print "LLaMA: (thinking...) "
  STDERR.flush

  # Prefill: process all prompt tokens at once
  x = SHAInet::SimpleMatrix.new(prompt_ids.size, d_model)
  prompt_ids.each_with_index { |id, i| d_model.times { |j| x[i, j] = emb.embeddings[id, j] } }
  net.transformer_layers.each { |l| x = l.as(SHAInet::LlamaBlock).forward_cached(x) }
  STDERR.print "\r" + " " * 40 + "\r"
  print "LLaMA: "

  # Generate tokens one at a time using cache
  eos_id = tokenizer.vocab["<|endoftext|>"]? || tokenizer.vocab["<|end_of_text|>"]? || tokenizer.vocab["</s>"]? || -1

  generated_ids = Array(Int32).new
  temperature = 0.8_f64
  repetition_penalty = 1.3_f64

  max_tokens.times do
    # Get logits from last position
    last = SHAInet::SimpleMatrix.new(1, d_model)
    d_model.times { |j| last[0, j] = x[x.rows - 1, j] }
    normed = fn.forward(last)
    logits = normed * w

    # Apply repetition penalty to recently generated tokens
    generated_ids.last(20).each do |prev_id|
      v = logits[0, prev_id]
      logits[0, prev_id] = v > 0 ? (v / repetition_penalty).to_f32 : (v * repetition_penalty).to_f32
    end

    # Temperature sampling with top-k
    vocab_size = logits.cols
    # Apply temperature and find top-40
    scored = Array(Tuple(Int32, Float64)).new(vocab_size)
    vocab_size.times do |j|
      next if j == eos_id
      scored << {j, logits[0, j] / temperature}
    end
    scored.sort_by! { |_, v| -v }
    top_k = scored.first(40)

    # Softmax over top-k
    max_val = top_k[0][1]
    exps = top_k.map { |id, v| {id, Math.exp(v - max_val)} }
    sum = exps.sum { |_, e| e }
    probs = exps.map { |id, e| {id, e / sum} }

    # Sample
    r = Random.rand
    cumulative = 0.0
    best_id = probs.last[0]
    probs.each do |id, p|
      cumulative += p
      if r <= cumulative
        best_id = id
        break
      end
    end

    break if best_id < 0
    generated_ids << best_id
    print tokenizer.decode([best_id])
    STDOUT.flush

    # Forward just the new token (O(1) with KV cache)
    x_new = SHAInet::SimpleMatrix.new(1, d_model)
    d_model.times { |j| x_new[0, j] = emb.embeddings[best_id, j] }
    net.transformer_layers.each { |l| x_new = l.as(SHAInet::LlamaBlock).forward_cached(x_new) }
    x = x_new
  end
  puts ""
  puts ""
end
