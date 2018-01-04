require "./spec_helper"

describe SHAInet::Layer do
  it "Initialize" do
    layer = SHAInet::Layer.new("memory", 4)
    layer.should be_a(SHAInet::Layer)
  end

  it "randomize seed" do
    layer = SHAInet::Layer.new("memory", 4)
    layer.random_seed
    (layer.neurons.sample.activation != 0.0).should eq(true)
  end

  it "check layer added functions" do
    layer = SHAInet::Layer.new("memory", 4)
    layer.type_change("eraser")
    layer.n_type.should eq("eraser")
  end
end
