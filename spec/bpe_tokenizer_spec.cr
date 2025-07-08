require "./spec_helper"

describe SHAInet::BPETokenizer do
  it "encodes and decodes text after training" do
    tokenizer = SHAInet::BPETokenizer.new
    tokenizer.train("hello world hello world", 30)
    encoded = tokenizer.encode("hello world")
    tokenizer.decode(encoded).should eq("hello world")
  end

  it "merges tokens correctly on long sequences" do
    tokenizer = SHAInet::BPETokenizer.new
    long_word = "ab" * 50
    tokenizer.train(long_word, 100)
    ids = tokenizer.encode(long_word)
    tokenizer.decode(ids).should eq(long_word)
  end
end
