require "../src/shainet"
require "json"

# LLaMA chat demo using Network#run with KV cache.
#
# Usage:
#   crystal run examples/llama_chat.cr -Denable_cuda
#   crystal run examples/llama_chat.cr -- /path/to/model-dir
#   crystal run examples/llama_chat.cr -- unsloth/Llama-3.2-1B-Instruct 512

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
max_tokens = (ARGV[1]? || "256").to_i

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
t = Time.instant
net = SHAInet::HFLoader.load_llama(model_dir)
STDERR.puts "Model loaded in #{(Time.instant - t).total_seconds.round(1)}s"
STDERR.puts "  Layers: #{net.transformer_layers.size}, d_model: #{net.transformer_layers.first.as(SHAInet::LlamaBlock).d_model}"

# --- Load tokenizer ---
tokenizer = SHAInet::BPETokenizer.from_hf(File.join(model_dir, "tokenizer.json"))
STDERR.puts "Tokenizer loaded (vocab: #{tokenizer.vocab.size})"
STDERR.puts ""

# --- Enable KV cache + GPU ---
net.use_kv_cache = true
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

  net.clear_cache!

  # Build prompt with LLaMA 3 chat template
  sp = ->(name : String) { tokenizer.vocab[name]? }
  bos = sp.call("<|begin_of_text|>")
  start_hdr = sp.call("<|start_header_id|>")
  end_hdr = sp.call("<|end_header_id|>")
  eot = sp.call("<|eot_id|>")

  prompt_ids = [] of Int32
  if bos && start_hdr && end_hdr && eot
    nl = tokenizer.encode("\n\n")
    prompt_ids << bos << start_hdr
    prompt_ids.concat(tokenizer.encode("user"))
    prompt_ids << end_hdr
    prompt_ids.concat(nl)
    prompt_ids.concat(tokenizer.encode(user_input.strip))
    prompt_ids << eot << start_hdr
    prompt_ids.concat(tokenizer.encode("assistant"))
    prompt_ids << end_hdr
    prompt_ids.concat(nl)
  else
    prompt_ids << (bos || 0)
    prompt_ids.concat(tokenizer.encode(user_input.strip))
  end

  STDERR.puts "(#{prompt_ids.size} tokens)"
  STDERR.print "LLaMA: (thinking...) "
  STDERR.flush

  # Prefill via net.run
  logits = net.run(prompt_ids, stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)

  STDERR.print "\r" + " " * 40 + "\r"
  print "LLaMA: "

  eot_id = tokenizer.vocab["<|eot_id|>"]? || -1
  eos_id = tokenizer.vocab["<|end_of_text|>"]? || -1
  generated_ids = Array(Int32).new
  temperature = 0.6_f64
  repetition_penalty = 1.2_f64
  gen_start = Time.instant

  max_tokens.times do
    vocab_size = logits.cols
    last_row = logits.rows - 1

    # Repetition penalty
    generated_ids.last(20).each do |prev_id|
      v = logits[last_row, prev_id]
      logits[last_row, prev_id] = v > 0 ? (v / repetition_penalty).to_f32 : (v * repetition_penalty).to_f32
    end

    # Temperature sampling with top-k
    scored = Array(Tuple(Int32, Float64)).new(vocab_size)
    vocab_size.times { |j| scored << {j, logits[last_row, j] / temperature} }
    scored.sort_by! { |_, v| v.nan? ? Float64::INFINITY : -v }
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

    break if best_id == eot_id || best_id == eos_id || best_id < 0
    break unless logits[last_row, best_id].finite?
    generated_ids << best_id
    print tokenizer.decode([best_id])
    STDOUT.flush

    # Decode next token via net.run (KV cache handles incremental state)
    logits = net.run([best_id], stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
  end

  gen_elapsed = (Time.instant - gen_start).total_seconds
  if generated_ids.size > 0
    STDERR.puts "(#{generated_ids.size} tokens, #{(generated_ids.size / gen_elapsed).round(1)} tok/s)"
  end
  puts ""
end
