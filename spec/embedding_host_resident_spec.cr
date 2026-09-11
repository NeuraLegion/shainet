require "./spec_helper"

# The embedding table is gather-only at inference, so #to_host! moves it to host
# RAM and releases the [vocab, l_size] device buffer. #embed must keep returning
# an identical CudaMatrix by gathering on the host and uploading just the rows
# asked for.
describe "EmbeddingLayer host residency" do
  it "matches the device-resident result after to_host!" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    vocab = 64
    dim = 16
    layer = SHAInet::EmbeddingLayer.new(vocab, dim)
    layer.host_resident?.should be_false

    ids = [0, 7, 7, 63, 1, 42]
    before = layer.embed(ids).to_simple

    layer.to_host!
    layer.host_resident?.should be_true
    layer.embeddings.should be_a(SHAInet::SimpleMatrix)

    after = layer.embed(ids).to_simple

    after.rows.should eq before.rows
    after.cols.should eq before.cols
    before.rows.times do |r|
      before.cols.times do |c|
        (after[r, c] - before[r, c]).abs.should be < 1e-6
      end
    end
  end

  it "is idempotent and keeps to_gpu! from pulling the table back" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    layer = SHAInet::EmbeddingLayer.new(32, 8)
    layer.to_host!
    layer.to_host! # second call must not raise or re-free
    layer.to_gpu!  # must not move the table back onto the device
    layer.host_resident?.should be_true
    layer.embeddings.should be_a(SHAInet::SimpleMatrix)
  end

  it "serves repeated and varying batch sizes from the host table" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    vocab = 48
    dim = 12
    layer = SHAInet::EmbeddingLayer.new(vocab, dim)
    reference = layer.embed_cpu((0...vocab).to_a)
    layer.to_host!

    [[3], [3, 4, 5], [9], [0, 47, 23, 23]].each do |ids|
      got = layer.embed(ids).to_simple
      got.rows.should eq ids.size
      ids.each_with_index do |id, row|
        dim.times do |c|
          (got[row, c] - reference[id, c]).abs.should be < 1e-6
        end
      end
    end
  end

  it "rejects out-of-range token ids instead of reading past the table" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    layer = SHAInet::EmbeddingLayer.new(16, 4)
    layer.to_host!
    expect_raises(IndexError) { layer.embed([16]) }
    expect_raises(IndexError) { layer.embed([-1]) }
  end
end
