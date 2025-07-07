require "./spec_helper"

describe "Embedding GPU parity" do
  it "matches CPU and GPU updates" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    # Create embedding layers with fixed values
    vocab_size = 3
    embed_size = 2
    lr = 0.1

    # First with GPU disabled
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    cpu_layer = SHAInet::EmbeddingLayer.new(vocab_size, embed_size)

    # Set all embeddings to 0.5
    vocab_size.times do |r|
      embed_size.times do |c|
        cpu_layer.embeddings[r, c] = 0.5
      end
    end

    # Apply gradients
    cpu_layer.embed(1)
    cpu_layer.gradients[1, 0] = 0.2
    cpu_layer.gradients[1, 1] = 0.2
    cpu_layer.apply_gradients(lr)

    # Now with GPU
    ENV.delete("SHAINET_DISABLE_CUDA")
    gpu_layer = SHAInet::EmbeddingLayer.new(vocab_size, embed_size)

    # Set all embeddings to 0.5
    vocab_size.times do |r|
      embed_size.times do |c|
        gpu_layer.embeddings[r, c] = 0.5
      end
    end

    # Make sure embeddings are synced to device
    if gpu_layer.embeddings.is_a?(SHAInet::CudaMatrix)
      gpu_layer.embeddings.as(SHAInet::CudaMatrix).sync_to_device!
    end

    # Apply gradients
    gpu_layer.embed(1)
    gpu_layer.gradients[1, 0] = 0.2
    gpu_layer.gradients[1, 1] = 0.2
    # Make sure gradients are synced to device
    if gpu_layer.gradients.is_a?(SHAInet::CudaMatrix)
      gpu_layer.gradients.as(SHAInet::CudaMatrix).sync_to_device!
    end
    gpu_layer.apply_gradients(lr)

    # Check results for updated values
    vocab_size.times do |r|
      embed_size.times do |c|
        gpu_val = gpu_layer.embeddings[r, c]
        cpu_val = cpu_layer.embeddings[r, c]
        gpu_val.should be_close(cpu_val, 1e-6)
      end
    end
  end
end
