require "./spec_helper"

describe SHAInet::BPETokenizer do
  it "encodes and decodes text after training" do
    tokenizer = SHAInet::BPETokenizer.new
    tokenizer.train("hello world hello world", 30)
    encoded = tokenizer.encode("hello world")
    tokenizer.decode(encoded).should eq("hello world")
  end

  it "trains using CUDA pair counting when available" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    tokenizer = SHAInet::BPETokenizer.new
    tokenizer.train("hello world hello world", 30)
    tokenizer.vocab.size.should be > 0
  end
end
