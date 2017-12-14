require "./spec_helper"

describe SHAInet do
  # TODO: Write tests
  it "test normalization to one-hot vectors" do
    payloads = ["abc", "12345", "!@#"]
    input_size, vocabulary_v, payloads_v = SHAInet.normalize_stcv(payloads)
    pp input_size
    puts "vocabulary_v is:"
    vocabulary_v.each { |x| puts x }
    puts "payloads_v is"
    payloads_v.each { |x| puts x }
  end
end
