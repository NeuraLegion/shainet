require "../src/shainet"
require "json"

# Simple LLaMA 3.2 chat demo
# Usage: crystal run examples/llama_chat.cr -- /path/to/model-dir

model_dir = ARGV[0]? || "/tmp/llama32"
max_tokens = (ARGV[1]? || "50").to_i

# --- Minimal HF BPE Tokenizer ---
class HFTokenizer
  @vocab : Hash(String, Int32)
  @id_to_token : Hash(Int32, String)
  @merges : Array(Tuple(String, String))
  @bos_id : Int32
  @eos_id : Int32

  def initialize(path : String)
    data = JSON.parse(File.read(path))
    model = data["model"]

    @vocab = Hash(String, Int32).new
    model["vocab"].as_h.each { |k, v| @vocab[k] = v.as_i }
    @id_to_token = Hash(Int32, String).new
    @vocab.each { |k, v| @id_to_token[v] = k }

    @merges = Array(Tuple(String, String)).new
    if merges = model["merges"]?
      merges.as_a.each do |m|
        parts = m.as_s.split(' ', 2)
        @merges << {parts[0], parts[1]} if parts.size == 2
      end
    end

    # Special tokens
    @bos_id = @vocab["<|begin_of_text|>"]? || @vocab["<s>"]? || 1
    @eos_id = @vocab["<|end_of_text|>"]? || @vocab["</s>"]? || 2
  end

  def bos_id : Int32
    @bos_id
  end

  def eos_id : Int32
    @eos_id
  end

  def encode(text : String) : Array(Int32)
    # Split into bytes represented as token strings (Ġ = space prefix in LLaMA 3)
    tokens = [] of String
    text.each_char_with_index do |ch, i|
      s = if i > 0 || text[0] == ' '
            ch == ' ' ? "Ġ" : ch.to_s
          else
            ch.to_s
          end
      # Handle space followed by char
      if ch == ' ' && i < text.size - 1
        tokens << "Ġ"
      else
        tokens << (i > 0 && text[i - 1] == ' ' ? "Ġ#{ch}" : ch.to_s)
      end
    end

    # Greedy BPE merge
    @merges.each do |pair|
      i = 0
      while i < tokens.size - 1
        if tokens[i] == pair[0] && tokens[i + 1] == pair[1]
          tokens[i] = pair[0] + pair[1]
          tokens.delete_at(i + 1)
        else
          i += 1
        end
      end
    end

    # Map to IDs (unknown → byte fallback)
    tokens.map { |t| @vocab[t]? || 0 }
  end

  def decode(ids : Array(Int32)) : String
    text = ids.map { |id| @id_to_token[id]? || "" }.join
    text.gsub("Ġ", " ")
  end

  def decode_token(id : Int32) : String
    (@id_to_token[id]? || "").gsub("Ġ", " ")
  end
end

# --- Load model ---
STDERR.puts "Loading model from #{model_dir}..."
t = Time.monotonic
net = SHAInet::HFLoader.load_llama(model_dir)
STDERR.puts "Model loaded in #{(Time.monotonic - t).total_seconds.round(1)}s"
STDERR.puts "  Layers: #{net.transformer_layers.size}, d_model: #{net.transformer_layers.first.as(SHAInet::LlamaBlock).d_model}"

# --- Load tokenizer ---
tokenizer = HFTokenizer.new(File.join(model_dir, "tokenizer.json"))
STDERR.puts "Tokenizer loaded (vocab: #{tokenizer.@vocab.size})"
STDERR.puts ""

# --- Chat loop ---
loop do
  STDERR.print "You: "
  user_input = gets
  break if user_input.nil? || user_input.strip.empty?

  # Encode with BOS
  prompt_ids = [tokenizer.bos_id] + tokenizer.encode(user_input.strip)
  STDERR.puts "(#{prompt_ids.size} tokens)"
  print "LLaMA: "

  # Generate
  ids = prompt_ids.dup
  max_tokens.times do
    # Build input matrix (column vector of token IDs)
    input = SHAInet::SimpleMatrix.new(ids.size, 1)
    ids.each_with_index { |id, i| input[i, 0] = id.to_f32 }

    # Forward pass — Network.run returns last-token logits [1, vocab]
    output = net.run(input)

    # Greedy argmax
    best_id = 0
    best_val = -Float64::INFINITY
    output.cols.times do |j|
      v = output[0, j]
      if v > best_val
        best_val = v
        best_id = j
      end
    end

    break if best_id == tokenizer.eos_id
    ids << best_id
    token_str = tokenizer.decode_token(best_id)
    print token_str
    STDOUT.flush
  end
  puts ""
  puts ""
end
