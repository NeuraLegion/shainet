require "./spec_helper"
require "../src/shainet"

describe "RoPE (half-split convention)" do
  it "is identity at position 0" do
    m = SHAInet::SimpleMatrix.new(1, 8)
    8.times { |j| m[0, j] = (j + 1).to_f32 }
    out = SHAInet::RoPE.apply(m, 0, 10000.0)
    8.times { |j| out[0, j].should be_close(m[0, j], 1e-5) }
  end

  it "rotates element i with element i+head_dim/2 (half-split)" do
    # head_dim=4, half=2. At position 1, first freq pair (i=0) uses freq=1.0
    head_dim = 4
    m = SHAInet::SimpleMatrix.new(1, head_dim)
    m[0, 0] = 1.0_f32 # x0
    m[0, 1] = 0.0_f32
    m[0, 2] = 0.0_f32 # x0's partner is index 0+half=2
    m[0, 3] = 0.0_f32
    out = SHAInet::RoPE.apply(m, 1, 10000.0)
    # freq for i=0: theta^0 = 1.0, angle = 1*1 = 1 rad
    # out[0] = x0*cos - x[2]*sin = 1*cos(1) - 0 = cos(1)
    # out[2] = x[2]*cos + x0*sin = 0 + 1*sin(1) = sin(1)
    out[0, 0].should be_close(Math.cos(1.0), 1e-4)
    out[0, 2].should be_close(Math.sin(1.0), 1e-4)
  end
end

describe "LlamaBlock KV cache" do
  it "produces same logits as full forward (cache consistency)" do
    model_dir = File.join(__DIR__, "test_data/tiny-llama")
    pending!("tiny-llama fixture missing") unless File.exists?(File.join(model_dir, "model.safetensors"))

    net = SHAInet::HFLoader.load_llama(model_dir)
    emb = net.hidden_layers.find(&.is_a?(SHAInet::EmbeddingLayer)).as(SHAInet::EmbeddingLayer)
    d = net.transformer_layers.first.as(SHAInet::LlamaBlock).d_model

    tokens = [1, 42, 100, 7]

    # Full forward
    full_x = SHAInet::SimpleMatrix.new(tokens.size, d)
    tokens.each_with_index { |t, i| d.times { |j| full_x[i, j] = emb.embeddings[t, j] } }
    net.transformer_layers.each { |l| full_x = l.as(SHAInet::LlamaBlock).forward(full_x) }

    # Cached: prefill first 3, then incremental last 1
    net.transformer_layers.each { |l| l.as(SHAInet::LlamaBlock).clear_cache! }
    pre = SHAInet::SimpleMatrix.new(3, d)
    tokens[0, 3].each_with_index { |t, i| d.times { |j| pre[i, j] = emb.embeddings[t, j] } }
    net.transformer_layers.each { |l| pre = l.as(SHAInet::LlamaBlock).forward_cached(pre) }

    inc = SHAInet::SimpleMatrix.new(1, d)
    d.times { |j| inc[0, j] = emb.embeddings[tokens[3], j] }
    net.transformer_layers.each { |l| inc = l.as(SHAInet::LlamaBlock).forward_cached(inc) }

    # Last row of full forward should match incremental output
    d.times do |j|
      inc[0, j].should be_close(full_x[tokens.size - 1, j], 1e-3)
    end
  end
end
