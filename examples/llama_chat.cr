require "../src/shainet"
require "json"

# LLaMA chat demo — runs a HuggingFace LLaMA model with KV cache.
#
# Usage:
#   crystal run examples/llama_chat.cr -Denable_cuda            # default: Llama-3.2-1B (auto-download)
#   crystal run examples/llama_chat.cr -- unsloth/Llama-3.2-1B  # specific HF repo
#   crystal run examples/llama_chat.cr -- /path/to/model-dir    # local model directory
#
# Models are cached under the system temp dir. Build with -Denable_cuda and
# set LD_LIBRARY_PATH to the kernels for GPU acceleration.

DEFAULT_REPO = "unsloth/Llama-3.2-1B-Instruct"
MODEL_FILES  = {"config.json", "model.safetensors", "tokenizer.json"}

def download_model(model_dir : String, repo : String)
  Dir.mkdir_p(model_dir)
  base_url = "https://huggingface.co/#{repo}/resolve/main"
  MODEL_FILES.each do |file|
    path = File.join(model_dir, file)
    next if File.exists?(path) && File.size(path) > 0
    STDERR.puts "  Downloading #{file}..."
    status = Process.run("curl", ["-fL", "--progress-bar", "#{base_url}/#{file}", "-o", path],
      output: Process::Redirect::Inherit, error: Process::Redirect::Inherit)
    raise "Failed to download #{file} from #{repo}" unless status.success?
  end
end

arg = ARGV[0]?
max_tokens = (ARGV[1]? || "60").to_i

# Resolve the model directory:
#  - existing local directory       -> use as-is
#  - HF repo ("org/name") or nil     -> download into temp cache
model_dir =
  if arg && Dir.exists?(arg)
    arg
  else
    repo = arg || DEFAULT_REPO
    dir = File.join(Dir.tempdir, "shainet_" + repo.gsub("/", "_"))
    unless File.exists?(File.join(dir, "model.safetensors"))
      STDERR.puts "Downloading #{repo}..."
      download_model(dir, repo)
    end
    dir
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

  # Build prompt using the LLaMA 3 chat template. Special tokens delimit the
  # user turn and open the assistant turn so the instruct model responds.
  sp = ->(name : String) { tokenizer.vocab[name]? }
  bos = sp.call("<|begin_of_text|>")
  start_hdr = sp.call("<|start_header_id|>")
  end_hdr = sp.call("<|end_header_id|>")
  eot = sp.call("<|eot_id|>")

  prompt_ids = [] of Int32
  if bos && start_hdr && end_hdr && eot
    nl = tokenizer.encode("\n\n")
    prompt_ids << bos
    prompt_ids << start_hdr
    prompt_ids.concat(tokenizer.encode("user"))
    prompt_ids << end_hdr
    prompt_ids.concat(nl)
    prompt_ids.concat(tokenizer.encode(user_input.strip))
    prompt_ids << eot
    prompt_ids << start_hdr
    prompt_ids.concat(tokenizer.encode("assistant"))
    prompt_ids << end_hdr
    prompt_ids.concat(nl)
  else
    # Base model fallback: just BOS + text
    prompt_ids << (bos || 0)
    prompt_ids.concat(tokenizer.encode(user_input.strip))
  end

  STDERR.puts "(#{prompt_ids.size} tokens)"
  STDERR.print "LLaMA: (thinking...) "
  STDERR.flush

  # Prefill: process all prompt tokens at once
  x = SHAInet::SimpleMatrix.new(prompt_ids.size, d_model)
  prompt_ids.each_with_index { |id, i| d_model.times { |j| x[i, j] = emb.embeddings[id, j] } }
  net.transformer_layers.each { |l| x = l.as(SHAInet::LlamaBlock).forward_cached(x) }
  STDERR.print "\r" + " " * 40 + "\r"
  print "LLaMA: "

  # Generate tokens one at a time using cache.
  # Stop on end-of-turn (<|eot_id|>) or end-of-text.
  eot_id = tokenizer.vocab["<|eot_id|>"]? || -1
  eos_id = tokenizer.vocab["<|end_of_text|>"]? || tokenizer.vocab["<|endoftext|>"]? || tokenizer.vocab["</s>"]? || -1

  generated_ids = Array(Int32).new
  temperature = 0.6_f64
  repetition_penalty = 1.2_f64

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

    # Temperature sampling with top-k (keep stop tokens as candidates)
    vocab_size = logits.cols
    scored = Array(Tuple(Int32, Float64)).new(vocab_size)
    vocab_size.times { |j| scored << {j, logits[0, j] / temperature} }
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

    # Stop on end-of-turn / end-of-text
    break if best_id == eot_id || best_id == eos_id || best_id < 0
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
