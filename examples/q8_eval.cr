require "../src/shainet"

# Q8 vs fp32 evaluation + benchmark for LLaMA 3.2 1B Instruct.
#
#   crystal run examples/q8_eval.cr --release -Denable_cuda -- q8     [model_dir] [gen]
#   crystal run examples/q8_eval.cr --release -Denable_cuda -- fp32   [model_dir] [gen]
#
# Greedy-decodes the "largest dog breed" prompt; expects output to start with
# "The largest dog breed is the Irish Wolfhound". Prints tok/s.

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

mode = (ARGV[0]? || "q8").downcase
arg = ARGV[1]?
gen_tokens = (ARGV[2]? || "24").to_i

model_dir =
  if arg && Dir.exists?(arg)
    arg
  else
    dir = File.join(Dir.tempdir, "shainet_" + DEFAULT_REPO.gsub("/", "_"))
    unless File.exists?(File.join(dir, "model.safetensors"))
      STDERR.puts "Downloading #{DEFAULT_REPO}..."
      download_model(dir, DEFAULT_REPO)
    end
    dir
  end

STDERR.puts "Loading model from #{model_dir} (mode=#{mode})..."
net = SHAInet::HFLoader.load_llama(model_dir, quantize: SHAInet::CUDA.fully_available? && mode == "q8")
tokenizer = SHAInet::BPETokenizer.from_hf(File.join(model_dir, "tokenizer.json"))

net.use_kv_cache = true
if SHAInet::CUDA.fully_available?
  if mode == "q8"
    STDERR.puts "Weights stream-quantized to Q8 during load."
  else
    STDERR.puts "Moving to GPU (fp32)..."
    net.transformer_layers.each { |l| l.as(SHAInet::LlamaBlock).to_gpu! }
  end
  if info = SHAInet::CUDA.memory_info
    used = (info[:total] - info[:free]) / 1024.0 / 1024.0
    STDERR.puts "VRAM total in use: #{used.round(1)} MB"
  end
end

# Build prompt with LLaMA 3 chat template.
sp = ->(name : String) { tokenizer.vocab[name]? }
bos = sp.call("<|begin_of_text|>").not_nil!
start_hdr = sp.call("<|start_header_id|>").not_nil!
end_hdr = sp.call("<|end_header_id|>").not_nil!
eot = sp.call("<|eot_id|>").not_nil!
nl = tokenizer.encode("\n\n")

prompt = "What's the largest dog breed?"
ids = [] of Int32
ids << bos << start_hdr
ids.concat(tokenizer.encode("user"))
ids << end_hdr
ids.concat(nl)
ids.concat(tokenizer.encode(prompt))
ids << eot << start_hdr
ids.concat(tokenizer.encode("assistant"))
ids << end_hdr
ids.concat(nl)

STDERR.puts "Prompt: #{prompt} (#{ids.size} tokens)"

eot_id = tokenizer.vocab["<|eot_id|>"]? || -1
eos_id = tokenizer.vocab["<|end_of_text|>"]? || -1

net.clear_cache!
logits = net.run(ids, stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)

generated = [] of Int32
out = String::Builder.new
t0 = Time.instant
gen_tokens.times do
  last_row = logits.rows - 1
  best_id = SHAInet::Sampler.greedy(logits, last_row)
  break if best_id == eot_id || best_id == eos_id
  generated << best_id
  out << tokenizer.decode([best_id])
  logits = net.run([best_id], stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
end
elapsed = (Time.instant - t0).total_seconds

puts ""
puts "Output: #{out.to_s}"
puts ""
toks = generated.size
printf("Decoded %d tokens in %.2fs = %.2f tok/s\n", toks, elapsed, toks / elapsed)
