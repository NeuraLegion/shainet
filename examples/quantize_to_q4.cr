require "../src/shainet"
require "json"

# Stream-convert a model's weights to a Q4 cache, one tensor at a time.
#
#   crystal run examples/quantize_to_q4.cr --release -Denable_cuda -- /path/to/model-dir
#
# Peak memory is ONE tensor: read -> quantize -> write -> free -> next. It cannot OOM regardless
# of model size, which is why it is a separate step from the agent's in-process load. The agent
# then loads the resulting <model-dir>/.q4 directory directly and never touches the bf16 originals.
#
# The layout matches what HFLoader.load_qwen35_from_cache expects: Q4 weights per layer keyed
# "layer.<i>.<name>", the embedding and small fp32 weights as raw binary, plus config/tokenizer.

abort "Usage: #{PROGRAM_NAME} <model-dir>" unless ARGV.size >= 1
model_dir = ARGV[0]
cache_dir = File.join(model_dir, ".q4")

if File.exists?(File.join(cache_dir, "manifest.json"))
  STDERR.puts "Already converted: #{cache_dir}"
  exit 0
end
abort "config.json not found in #{model_dir}" unless File.exists?(File.join(model_dir, "config.json"))

config = SHAInet::HFLoader.load_llama_config(File.join(model_dir, "config.json"))
types = config.layer_types || SHAInet::HFLoader.default_layer_types(config.num_hidden_layers)
d = config.hidden_size
k_dim = config.linear_num_key_heads * config.linear_key_head_dim
v_dim = config.linear_num_value_heads * config.linear_value_head_dim
q_dim = config.num_attention_heads * (config.head_dim || (d // config.num_attention_heads))

Dir.mkdir_p(cache_dir)
manifest = Hash(String, Hash(String, Int64)).new
sf = SHAInet::HFLoader.open_safetensors(model_dir)
t0 = Time.monotonic

# Quantize a [in, out] SimpleMatrix to Q4 and write it under `key`, then let it be collected.
write_q4 = ->(key : String, w : SHAInet::SimpleMatrix) do
  q, dsc, sub = SHAInet::Q4CudaMatrix.pack(w)
  base = key.gsub(".", "__")
  File.open(File.join(cache_dir, "#{base}.q"), "w") { |f| f.write(q.to_unsafe.to_slice(q.size)) }
  File.open(File.join(cache_dir, "#{base}.d"), "w") { |f| f.write(dsc.to_unsafe.as(Pointer(UInt8)).to_slice(dsc.size * 4)) }
  File.open(File.join(cache_dir, "#{base}.sub"), "w") { |f| f.write(sub.to_unsafe.to_slice(sub.size)) }
  manifest[key] = {"rows" => w.rows.to_i64, "cols" => w.cols.to_i64,
                   "q_size" => q.size.to_i64, "d_size" => dsc.size.to_i64, "sub_size" => sub.size.to_i64}
end

save_bin = ->(path : String, w : SHAInet::SimpleMatrix) do
  # Write in chunks: w.rows * w.cols * 4 overflows Int32 for the 27B embedding (5.08 GB).
  total = w.rows.to_i64 * w.cols.to_i64
  ptr = w.data.to_unsafe.as(Pointer(UInt8))
  File.open(path, "w") do |f|
    written = 0_i64
    byte_total = total * 4
    while written < byte_total
      n = Math.min(byte_total - written, 1_073_741_824_i64).to_i32
      f.write(Slice.new(ptr + written, n))
      written += n
    end
  end
end

# Embedding (raw fp32).
STDERR.puts "embedding..."
emb = sf.read_matrix("model.language_model.embed_tokens.weight")
save_bin.call(File.join(cache_dir, "embedding.bin"), emb)
emb.data.clear
GC.collect

types.each_with_index do |t, idx|
  pre = "model.language_model.layers.#{idx}."
  dir = File.join(cache_dir, "layer_#{idx}")
  Dir.mkdir_p(dir)
  STDERR.print "\rlayer #{idx + 1}/#{types.size}"
  STDERR.flush

  if t == "linear_attention"
    qkv = sf.read_matrix_transposed("#{pre}linear_attn.in_proj_qkv.weight")
    wq, wk, wv = SHAInet::HFLoader.split_qkv_contiguous(qkv, k_dim, v_dim)
    qkv.data.clear
    write_q4.call("layer.#{idx}.w_q", wq)
    write_q4.call("layer.#{idx}.w_k", wk)
    write_q4.call("layer.#{idx}.w_v", wv)
    write_q4.call("layer.#{idx}.w_gate", sf.read_matrix_transposed("#{pre}linear_attn.in_proj_z.weight"))
    write_q4.call("layer.#{idx}.w_o", sf.read_matrix_transposed("#{pre}linear_attn.out_proj.weight"))
    save_bin.call(File.join(dir, "w_alpha.bin"), sf.read_matrix_transposed("#{pre}linear_attn.in_proj_a.weight"))
    save_bin.call(File.join(dir, "w_beta.bin"), sf.read_matrix_transposed("#{pre}linear_attn.in_proj_b.weight"))
    save_bin.call(File.join(dir, "a_log.bin"), sf.read_matrix("#{pre}linear_attn.A_log"))
    save_bin.call(File.join(dir, "dt_bias.bin"), sf.read_matrix("#{pre}linear_attn.dt_bias"))
    save_bin.call(File.join(dir, "conv.bin"), sf.read_matrix("#{pre}linear_attn.conv1d.weight"))
    save_bin.call(File.join(dir, "out_norm.bin"), sf.read_matrix("#{pre}linear_attn.norm.weight"))
    save_bin.call(File.join(dir, "norm1.bin"), sf.read_matrix("#{pre}input_layernorm.weight"))
    save_bin.call(File.join(dir, "norm2.bin"), sf.read_matrix("#{pre}post_attention_layernorm.weight"))
  else
    qp = sf.read_matrix_transposed("#{pre}self_attn.q_proj.weight")
    if qp.cols == 2 * q_dim
      wq, wg = SHAInet::HFLoader.split_head_interleaved(qp, config.num_attention_heads, config.head_dim || (d // config.num_attention_heads))
      write_q4.call("layer.#{idx}.w_q", wq)
      write_q4.call("layer.#{idx}.w_gate_attn", wg)
    else
      write_q4.call("layer.#{idx}.w_q", qp)
    end
    write_q4.call("layer.#{idx}.w_k", sf.read_matrix_transposed("#{pre}self_attn.k_proj.weight"))
    write_q4.call("layer.#{idx}.w_v", sf.read_matrix_transposed("#{pre}self_attn.v_proj.weight"))
    write_q4.call("layer.#{idx}.w_o", sf.read_matrix_transposed("#{pre}self_attn.o_proj.weight"))
    save_bin.call(File.join(dir, "q_norm.bin"), sf.read_matrix("#{pre}self_attn.q_norm.weight"))
    save_bin.call(File.join(dir, "k_norm.bin"), sf.read_matrix("#{pre}self_attn.k_norm.weight"))
    save_bin.call(File.join(dir, "norm1.bin"), sf.read_matrix("#{pre}input_layernorm.weight"))
    save_bin.call(File.join(dir, "norm2.bin"), sf.read_matrix("#{pre}post_attention_layernorm.weight"))
  end
  # FFN
  write_q4.call("layer.#{idx}.ffn.gate", sf.read_matrix_transposed("#{pre}mlp.gate_proj.weight"))
  write_q4.call("layer.#{idx}.ffn.up", sf.read_matrix_transposed("#{pre}mlp.up_proj.weight"))
  write_q4.call("layer.#{idx}.ffn.down", sf.read_matrix_transposed("#{pre}mlp.down_proj.weight"))
  GC.collect
end
STDERR.puts ""

# Final norm and lm_head.
save_bin.call(File.join(cache_dir, "final_norm.bin"), sf.read_matrix("model.language_model.norm.weight"))
head_name = config.tie_word_embeddings ? "model.language_model.embed_tokens.weight" : "lm_head.weight"
write_q4.call("lm_head", sf.read_matrix_transposed(head_name))
GC.collect

# Config and tokenizer so the cache is standalone.
["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"].each do |fname|
  src = File.join(model_dir, fname)
  File.copy(src, File.join(cache_dir, fname)) if File.exists?(src)
end

File.write(File.join(cache_dir, "manifest.json"), manifest.to_json)
sf.close
dt = (Time.monotonic - t0).total_seconds
total = Dir.glob(File.join(cache_dir, "**/*")).select { |f| File.file?(f) }.sum { |f| File.size(f) }
STDERR.puts "Done: #{manifest.size} Q4 tensors in #{dt.round(1)}s, #{(total / 1_073_741_824.0).round(2)} GiB"
STDERR.puts "Load with: ./agent #{cache_dir}"
