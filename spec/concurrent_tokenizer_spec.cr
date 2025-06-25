require "./spec_helper"

describe SHAInet::ConcurrentTokenizer(SHAInet::Tokenizer) do
  it "encodes text concurrently" do
    tokenizer = SHAInet::Tokenizer.new
    tokenizer.build("hello world test concurrency")
    concurrent = SHAInet::ConcurrentTokenizer(SHAInet::Tokenizer).new(tokenizer, 2)
    inputs = ["hello world", "test concurrency"]

    sequential = inputs.map { |txt| tokenizer.encode(txt) }
    parallel = concurrent.encode_batch(inputs)
    parallel.should eq(sequential)
  end
end
