require "../src/shainet"
require "json"

# LLaMA chat demo using Network#run with KV cache.
#
# Usage:
#   crystal run examples/llama_chat.cr -Denable_cuda
#   crystal run examples/llama_chat.cr -- /path/to/model-dir
#   crystal run examples/llama_chat.cr -- unsloth/Llama-3.2-1B-Instruct 512

DEFAULT_REPO = "unsloth/Llama-3.2-1B-Instruct"

def download_model(model_dir : String, repo : String)
  Dir.mkdir_p(model_dir)
  base_url = "https://huggingface.co/#{repo}/resolve/main"

  fetch = ->(file : String, required : Bool) : Bool {
    path = File.join(model_dir, file)
    return true if File.exists?(path) && File.size(path) > 0
    STDERR.puts "  Downloading #{file}..."
    status = Process.run("curl", ["-fL", "--progress-bar", "#{base_url}/#{file}", "-o", path],
      output: Process::Redirect::Inherit, error: Process::Redirect::Inherit)
    File.delete(path) if !status.success? && File.exists?(path)
    raise "Failed to download #{file} from #{repo}" if required && !status.success?
    status.success?
  }

  fetch.call("config.json", true)
  fetch.call("tokenizer.json", true)

  # Weights: single model.safetensors if it exists, otherwise the sharded set
  # described by model.safetensors.index.json.
  return if fetch.call("model.safetensors", false)
  fetch.call("model.safetensors.index.json", true)
  index = JSON.parse(File.read(File.join(model_dir, "model.safetensors.index.json")))
  shards = index["weight_map"].as_h.values.map(&.as_s).uniq!
  STDERR.puts "  Sharded model: #{shards.size} shard(s)"
  shards.each { |shard| fetch.call(shard, true) }
end

def model_complete?(dir : String) : Bool
  single = File.join(dir, "model.safetensors")
  return true if File.exists?(single) && File.size(single) > 0
  index = File.join(dir, "model.safetensors.index.json")
  return false unless File.exists?(index) && File.size(index) > 0
  # Sharded: every shard named in the index must exist and be non-empty.
  shards = JSON.parse(File.read(index))["weight_map"].as_h.values.map(&.as_s).uniq!
  shards.all? { |s| (p = File.join(dir, s)) && File.exists?(p) && File.size(p) > 0 }
end

arg = ARGV[0]?
max_tokens = (ARGV[1]? || "256").to_i

model_dir =
  if arg && Dir.exists?(arg)
    arg
  else
    repo = arg || DEFAULT_REPO
    dir = File.join(Dir.tempdir, "shainet_" + repo.gsub("/", "_"))
    # download_model is idempotent (it skips already-complete files), so calling
    # it whenever the model is incomplete self-heals partial/interrupted runs.
    unless model_complete?(dir)
      STDERR.puts "Downloading #{repo}..."
      download_model(dir, repo)
    end
    dir
  end

# --- Load model ---
STDERR.puts "Loading model from #{model_dir}..."
t = Time.instant
# Stream-quantize to Q8 during load (CUDA only, unless SHAINET_FP32=1) so large
# models never materialize their full fp32 weights in host RAM at once.
quantize = SHAInet::CUDA.fully_available? && !ENV["SHAINET_FP32"]?
bits = ENV["SHAINET_Q4"]? ? 4 : 8
offload = ENV.fetch("SHAINET_MOE_OFFLOAD", "0") == "1"
mode = ENV["SHAINET_FP32"]? ? "fp32" : "Q#{bits}"
STDERR.puts "  Mode: #{mode}#{offload ? " (MoE experts offloaded to host RAM)" : ""}"
begin
  net = SHAInet::HFLoader.load(model_dir, quantize: quantize, bits: bits)
rescue ex
  if (ex.message.try(&.downcase.includes?("memory"))) && !offload
    STDERR.puts ""
    STDERR.puts "GPU ran out of memory loading this model. If it is a large Mixture-of-Experts"
    STDERR.puts "model (e.g. Qwen3-MoE), retry with experts offloaded to host RAM:"
    STDERR.puts "  SHAINET_Q4=1 SHAINET_MOE_OFFLOAD=1 #{PROGRAM_NAME} #{model_dir}"
  end
  raise ex
end
STDERR.puts "Model loaded in #{(Time.instant - t).total_seconds.round(1)}s"
STDERR.puts "  Layers: #{net.transformer_layers.size}, d_model: #{net.transformer_layers.first.as(SHAInet::LlamaBlock).d_model}"

# --- Load tokenizer ---
tokenizer = SHAInet::BPETokenizer.from_hf(File.join(model_dir, "tokenizer.json"))
STDERR.puts "Tokenizer loaded (vocab: #{tokenizer.vocab.size})"
STDERR.puts ""

# --- Enable KV cache + GPU ---
net.use_kv_cache = true
if SHAInet::CUDA.fully_available?
  if ENV["SHAINET_FP32"]?
    STDERR.puts "Moving to GPU (fp32)..."
    net.transformer_layers.each { |l| l.as(SHAInet::LlamaBlock).to_gpu! }
  else
    # Weights were already stream-quantized (Q8 or Q4) during load.
    if info = SHAInet::CUDA.memory_info
      STDERR.puts "  Quantized to Q#{bits}; VRAM in use: #{((info[:total] - info[:free]) / 1024.0 / 1024.0).round(1)} MB"
    end
  end
  STDERR.puts "GPU ready!"
end

# --- Chat loop ---
loop do
  STDERR.print "You: "
  user_input = gets
  break if user_input.nil? || user_input.strip.empty?

  net.clear_cache!

  # Build prompt using whichever chat template the tokenizer supports.
  sp = ->(name : String) { tokenizer.vocab[name]? }
  bos = sp.call("<|begin_of_text|>")
  start_hdr = sp.call("<|start_header_id|>")
  end_hdr = sp.call("<|end_header_id|>")
  eot = sp.call("<|eot_id|>")
  im_start = sp.call("<|im_start|>")
  im_end = sp.call("<|im_end|>")

  prompt_ids = [] of Int32
  if im_start && im_end
    # Qwen / ChatML: <|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
    nl = tokenizer.encode("\n")
    prompt_ids << im_start
    prompt_ids.concat(tokenizer.encode("user"))
    prompt_ids.concat(nl)
    prompt_ids.concat(tokenizer.encode(user_input.strip))
    prompt_ids << im_end
    prompt_ids.concat(nl)
    prompt_ids << im_start
    prompt_ids.concat(tokenizer.encode("assistant"))
    prompt_ids.concat(nl)
  elsif bos && start_hdr && end_hdr && eot
    # LLaMA 3 chat template
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

  # Stop on whichever end-of-turn / end-of-text tokens this model defines.
  stop_ids = [] of Int32
  ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"].each do |name|
    if id = tokenizer.vocab[name]?
      stop_ids << id unless stop_ids.includes?(id)
    end
  end
  generated_ids = Array(Int32).new
  sampler = SHAInet::Sampler.new(temperature: 0.6, top_k: 40, repetition_penalty: 1.2)
  gen_start = Time.instant

  max_tokens.times do
    last_row = logits.rows - 1

    # Penalize recently emitted tokens, then temperature + top-k sample.
    sampler.apply_repetition_penalty!(logits, generated_ids, window: 20, row: last_row)
    best_id = sampler.sample(logits, last_row)

    break if best_id < 0 || stop_ids.includes?(best_id)
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
    if offload && SHAInet::CUDA.fully_available?
      cs = SHAInet::Q4HostMatrix.cache_stats
      STDERR.puts "  [expert cache] #{(cs[:hit_rate]*100).round(1)}% hit, #{cs[:resident]} resident, #{cs[:used_mb].round(0)}/#{cs[:budget_mb].round(0)} MB"
    end
  end
  puts ""
end
