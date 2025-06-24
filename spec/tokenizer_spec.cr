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

  it "encodes to a matrix using CUDA when available" do
    tokenizer = SHAInet::Tokenizer.new
    tokenizer.build("hello world")
    matrix = tokenizer.encode_matrix("hello world")
    if SHAInet::CUDA.available?
      matrix.should be_a(SHAInet::CudaMatrix)
    else
      matrix.should be_a(SHAInet::SimpleMatrix)
    end
    matrix[0, 0].should eq(0.0)
    matrix[0, 1].should eq(1.0)
  end
end
