require "./spec_helper"
require "../src/shainet"

describe "HFLoader LLaMA" do
  it "loads tiny-llama from safetensors" do
    model_dir = File.join(__DIR__, "test_data/tiny-llama")
    pending!("tiny-llama fixture missing") unless File.exists?(File.join(model_dir, "model.safetensors"))

    model = SHAInet::HFLoader.load_llama(model_dir)

    model.blocks.size.should eq(2)
    model.config.hidden_size.should eq(16)
    model.config.num_attention_heads.should eq(4)
    model.config.vocab_size.should eq(32000)
  end

  it "runs forward pass and produces logits" do
    model_dir = File.join(__DIR__, "test_data/tiny-llama")
    pending!("tiny-llama fixture missing") unless File.exists?(File.join(model_dir, "model.safetensors"))

    model = SHAInet::HFLoader.load_llama(model_dir)

    # Forward pass with 3 tokens
    output = model.forward([1, 42, 100])

    # Output should be [3, 32000] (seq_len, vocab_size)
    output.rows.should eq(3)
    output.cols.should eq(32000)

    # Should not be all zeros
    has_nonzero = (0...output.cols).any? { |c| output[0, c] != 0.0 }
    has_nonzero.should be_true

    # All values finite
    all_finite = (0...3).all? do |r|
      (0...output.cols).all? { |c| output[r, c].finite? }
    end
    all_finite.should be_true
  end
end
