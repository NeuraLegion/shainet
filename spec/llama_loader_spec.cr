require "./spec_helper"
require "../src/shainet"

describe "HFLoader LLaMA" do
  it "loads tiny-llama into Network" do
    model_dir = File.join(__DIR__, "test_data/tiny-llama")
    pending!("tiny-llama fixture missing") unless File.exists?(File.join(model_dir, "model.safetensors"))

    net = SHAInet::HFLoader.load_llama(model_dir)
    net.should be_a(SHAInet::Network)
    net.transformer_layers.size.should eq(2)
    net.transformer_layers.first.should be_a(SHAInet::LlamaLayer)
  end

  it "runs forward pass via Network.run" do
    model_dir = File.join(__DIR__, "test_data/tiny-llama")
    pending!("tiny-llama fixture missing") unless File.exists?(File.join(model_dir, "model.safetensors"))

    net = SHAInet::HFLoader.load_llama(model_dir)

    # Input: 3 tokens as column vector
    input = SHAInet::SimpleMatrix.new(3, 1)
    input[0, 0] = 1.0
    input[1, 0] = 42.0
    input[2, 0] = 100.0

    output = net.run(input)

    # Network.run returns last-token logits [1, vocab_size] (standard LM behavior)
    output.rows.should eq(1)
    output.cols.should eq(32000)

    # Not all zeros
    has_nonzero = (0...output.cols).any? { |c| output[0, c] != 0.0 }
    has_nonzero.should be_true

    # All finite
    all_finite = (0...output.cols).all? { |c| output[0, c].finite? }
    all_finite.should be_true
  end

  it "loads via generic Network.load_from_hf" do
    model_dir = File.join(__DIR__, "test_data/tiny-llama")
    pending!("tiny-llama fixture missing") unless File.exists?(File.join(model_dir, "model.safetensors"))

    net = SHAInet::Network.load_from_hf(model_dir)
    net.should be_a(SHAInet::Network)
    net.transformer_layers.size.should eq(2)
  end
end
