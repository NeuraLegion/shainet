require "./spec_helper"

describe SHAInet::EmbeddingLayer do
  it "returns consistent embeddings" do
    layer = SHAInet::EmbeddingLayer.new(5, 4)
    first = layer.embed(1)
    second = layer.embed(1)
    first.should eq(second)
    activations = Array(Float64).new(layer.l_size) { |i| layer.activations[0, i] }
    activations.should eq(first)
  end

  it "updates embeddings during training" do
    # Create a layer directly to test embedding updates
    layer = SHAInet::EmbeddingLayer.new(3, 2)

    # Set some specific values for testing
    layer.embeddings[1, 0] = 0.5
    layer.embeddings[1, 1] = 0.5

    before = layer.lookup(1)
    puts "Before: #{before.inspect}, embeddings at 1: #{layer.embeddings[1, 0]}, #{layer.embeddings[1, 1]}"

    # Simulate embedding and gradient accumulation
    layer.embed(1)

    # Set gradients directly
    layer.gradients[1, 0] = 0.1
    layer.gradients[1, 1] = 0.1

    # Apply gradients with a learning rate
    layer.apply_gradients(0.1)

    after = layer.lookup(1)
    puts "After: #{after.inspect}, embeddings at 1: #{layer.embeddings[1, 0]}, #{layer.embeddings[1, 1]}"

    after.should_not eq(before)
  end
end
