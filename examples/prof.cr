require "../src/shainet"

model_dir = ARGV[0]? || "/tmp/llama32"
tok = SHAInet::BPETokenizer.from_hf(File.join(model_dir, "tokenizer.json"))
net = SHAInet::HFLoader.load_llama(model_dir)
net.transformer_layers.each { |l| l.as(SHAInet::LlamaBlock).to_gpu! }
emb = net.hidden_layers.find(&.is_a?(SHAInet::EmbeddingLayer)).as(SHAInet::EmbeddingLayer)
fn = net.final_norm.not_nil!
w_sm = net.output_layers.first.weights.as(SHAInet::SimpleMatrix)
w = w_sm.to_cuda.tap(&.mark_device_dirty!)
d = emb.embeddings.cols

ids = [128000] + tok.encode("The capital of France is")
net.transformer_layers.each { |l| l.as(SHAInet::LlamaBlock).clear_cache! }
x = SHAInet::SimpleMatrix.new(ids.size, d)
ids.each_with_index { |id, i| d.times { |j| x[i, j] = emb.embeddings[id, j] } }
net.transformer_layers.each { |l| x = l.as(SHAInet::LlamaBlock).forward_cached(x) }

n = 10
t = Time.monotonic
n.times do
  last = SHAInet::SimpleMatrix.new(1, d)
  d.times { |j| last[0, j] = x[x.rows - 1, j] }
  normed = fn.forward(last)
  n_gpu = SHAInet::CudaMatrix.new(normed.rows, normed.cols)
  n_gpu.raw_data.to_unsafe.copy_from(normed.data.to_unsafe, normed.rows * normed.cols)
  n_gpu.sync_to_device!("lm_head")
  logits_gpu = n_gpu * w
  logits_gpu.sync_from_device!("lm_head") if logits_gpu.device_dirty?
  top = (0...logits_gpu.cols).max_by { |j| v = logits_gpu.raw_data[j]; v.nan? ? -1e30_f32 : v }
  xn = SHAInet::SimpleMatrix.new(1, d)
  d.times { |j| xn[0, j] = emb.embeddings[top, j] }
  net.transformer_layers.each { |l| xn = l.as(SHAInet::LlamaBlock).forward_cached(xn) }
  x = xn
end
elapsed = (Time.monotonic - t).total_seconds
puts "#{n} tokens: #{elapsed.round(2)}s -> #{(elapsed / n).round(3)}s/token"
