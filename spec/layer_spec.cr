require "./spec_helper"

describe SHAInet::Layer do
  it "Initialize layer" do
    puts "\n"
    layer = SHAInet::Layer.new("memory", 4)
    layer.should be_a(SHAInet::Layer)
  end
end
