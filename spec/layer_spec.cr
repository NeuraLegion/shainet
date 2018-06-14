require "./spec_helper"

describe SHAInet::Layer do
  puts "############################################################"
  it "Initialize layer" do
    puts "\n"
    layer = SHAInet::Layer.new("memory", 4)
    layer.should be_a(SHAInet::Layer)
  end

  puts "############################################################"
  it "randomize seed for layer" do
    puts "\n"
    layer = SHAInet::Layer.new("memory", 4)
    layer.random_seed
    (layer.neurons.sample.activation != 0.0).should eq(true)
  end

  puts "############################################################"
  it "check layer added functions" do
    puts "\n"
    layer = SHAInet::Layer.new("memory", 4)
    layer.type_change("eraser")
    layer.n_type.should eq("eraser")
  end
end
