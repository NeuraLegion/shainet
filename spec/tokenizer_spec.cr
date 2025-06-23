require "./spec_helper"

describe SHAInet::Tokenizer do
  it "builds vocabulary and encodes/decodes" do
    tokenizer = SHAInet::Tokenizer.new
    tokenizer.build("hello world hello")
    tokenizer.vocab.size.should eq(2)
    ids = tokenizer.encode("hello world")
    ids.should eq([0, 1])
    tokenizer.decode(ids).should eq(["hello", "world"])
  end
end
