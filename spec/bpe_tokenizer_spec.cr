require "./spec_helper"

describe SHAInet::BPETokenizer do
  it "encodes and decodes text after training" do
    tokenizer = SHAInet::BPETokenizer.new
    tokenizer.train("hello world hello world", 30)
    encoded = tokenizer.encode("hello world")
    tokenizer.decode(encoded).should eq("hello world")
  end
end
