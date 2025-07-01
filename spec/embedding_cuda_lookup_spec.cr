require "./spec_helper"

describe "Embedding GPU lookup" do
  it "retrieves embeddings on the device" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?

    # Create a simple embedding layer with fixed values
    layer = SHAInet::EmbeddingLayer.new(5, 4)

    # Set the embedding values
    token_id = 1
    expected_values = [0.1, 0.2, 0.3, 0.4]

    # Set the values directly in the embeddings matrix
    expected_values.each_with_index do |val, idx|
      layer.embeddings[token_id, idx] = val
    end

    # Sync to device if using CUDA
    if layer.embeddings.is_a?(SHAInet::CudaMatrix)
      layer.embeddings.as(SHAInet::CudaMatrix).sync_to_device!
    end

    # Get embedding vector
    result = layer.lookup(token_id)

    # Compare results
    result.size.should eq(expected_values.size)
    result.each_with_index do |val, idx|
      val.should be_close(expected_values[idx], 1e-6)
    end
  end
end
