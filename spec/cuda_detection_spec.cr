require "./spec_helper"

describe "CUDA availability" do
  it "returns a boolean" do
    value = SHAInet::CUDA.available?
    value.should be_a(Bool)
  end

  it "reports version when available" do
    version = SHAInet::CUDA.version
    if version
      version.should be > 0
    else
      version.should be_nil
    end
  end
end
