require "./spec_helper"

# The workspace pool hands out reusable device buffers. Its one hard invariant is that
# a matrix in the pool is owned by NOBODY: whatever is in there can be popped and
# zeroed at any moment. Breaking that gives two live owners of one device buffer,
# which showed up as an intermittent SIGSEGV on the training path rather than as a
# wrong number, because the second owner zeroes or frees the first owner's memory.
private def pool_ready?
  SHAInet::CUDA.fully_available?
end

describe "CudaMatrix workspace pool ownership" do
  it "never pools the same matrix twice" do
    pending! "CUDA not available" unless pool_ready?

    SHAInet::CudaMatrix.clear_workspace_pool
    m = SHAInet::CudaMatrix.get_workspace(4, 4, "spec")
    SHAInet::CudaMatrix.return_workspace(m)
    SHAInet::CudaMatrix.return_workspace(m)

    # Two pops must not both yield the same object, which is what a double-pool does.
    a = SHAInet::CudaMatrix.get_workspace(4, 4, "spec")
    b = SHAInet::CudaMatrix.get_workspace(4, 4, "spec")
    a.same?(b).should be_false
  ensure
    SHAInet::CudaMatrix.clear_workspace_pool
  end

  it "does not pool the activation matrix it returns from an identity layer" do
    pending! "CUDA not available" unless pool_ready?

    # The identity activation path adopts the forward workspace as @activations and
    # returns it. Before the fix the ensure block also handed that exact object back
    # to the pool, so the caller's activations were sitting in the free list: the next
    # get_workspace of the same shape popped and zero!d them mid-training.
    layer = SHAInet::MatrixLayer.new(3, 3, SHAInet.none)
    layer.to_gpu!

    input = SHAInet::CudaMatrix.new(1, 3)
    3.times { |j| input[0, j] = 0.5 + j }
    input.mark_host_modified!
    input.sync_to_device!("spec_in")

    SHAInet::CudaMatrix.clear_workspace_pool
    act = layer.forward(input)

    # Asking the pool for the same shape must not return the live activations.
    reused = SHAInet::CudaMatrix.get_workspace(act.rows, act.cols, "spec")
    reused.same?(act).should be_false
  ensure
    SHAInet::CudaMatrix.clear_workspace_pool
  end

  it "keeps identity-layer activations intact after a same-shape workspace request" do
    pending! "CUDA not available" unless pool_ready?

    layer = SHAInet::MatrixLayer.new(3, 3, SHAInet.none)
    layer.to_gpu!

    input = SHAInet::CudaMatrix.new(1, 3)
    3.times { |j| input[0, j] = 1.0 + j }
    input.mark_host_modified!
    input.sync_to_device!("spec_in")

    SHAInet::CudaMatrix.clear_workspace_pool
    act = layer.forward(input)
    act.sync_from_device!("spec_out") if act.device_dirty?
    before = Array(Float64).new(act.cols) { |j| act[0, j] }

    # get_workspace zeroes whatever it pops. If the activations were pooled, this call
    # wipes them, so comparing before/after is a direct test of the invariant rather
    # than of object identity.
    scratch = SHAInet::CudaMatrix.get_workspace(act.rows, act.cols, "spec")
    scratch.zero!

    act.sync_from_device!("spec_recheck")
    act.cols.times { |j| act[0, j].should be_close(before[j], 1e-6) }
  ensure
    SHAInet::CudaMatrix.clear_workspace_pool
  end
end
