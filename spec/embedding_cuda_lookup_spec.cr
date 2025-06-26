require "./spec_helper"

describe "Embedding GPU lookup" do
  it "retrieves embeddings on the device" do
    pending! "CUDA not available" unless SHAInet::CUDA.available?
    layer = SHAInet::EmbeddingLayer.new(5, 4)
    ids = [1, 3]
    matrix = layer.embed(ids)
    matrix.should be_a(SHAInet::CudaMatrix)
    matrix.as(SHAInet::CudaMatrix).sync_from_device!
    ids.each_with_index do |id, row|
      layer.lookup(id).each_with_index do |val, col|
        matrix[row, col].should be_close(val, 1e-6)
      end
    end
  end
end
