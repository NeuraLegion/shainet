require "../src/shainet"
net = SHAInet::HFLoader.load_llama("/tmp/smollm")
emb = net.hidden_layers.find(&.is_a?(SHAInet::EmbeddingLayer)).as(SHAInet::EmbeddingLayer)
fn = net.final_norm.not_nil!; w = net.output_layers.first.weights.as(SHAInet::SimpleMatrix); d = 576

# Full forward [0, 5588, 28]
input = SHAInet::SimpleMatrix.new(3, 1)
[0, 5588, 28].each_with_index { |id, i| input[i, 0] = id.to_f32 }
full = net.run(input)
full_top = (1...full.cols).max_by { |j| full[0, j] }
puts "Full: #{full_top}"

# Cached: prefill [0, 5588], then add [28]
net.transformer_layers.each { |l| l.as(SHAInet::LlamaBlock).clear_cache! }
x = SHAInet::SimpleMatrix.new(2, d)
[0, 5588].each_with_index { |id, i| d.times { |j| x[i, j] = emb.embeddings[id, j] } }
net.transformer_layers.each { |l| x = l.as(SHAInet::LlamaBlock).forward_cached(x) }

x_new = SHAInet::SimpleMatrix.new(1, d)
d.times { |j| x_new[0, j] = emb.embeddings[28, j] }
net.transformer_layers.each { |l| x_new = l.as(SHAInet::LlamaBlock).forward_cached(x_new) }
last = SHAInet::SimpleMatrix.new(1, d)
d.times { |j| last[0, j] = x_new[0, j] }
logits = fn.forward(last) * w
cached_top = (1...logits.cols).max_by { |j| logits[0, j] }
puts "Cached: #{cached_top}"
puts "Match: #{full_top == cached_top}"
