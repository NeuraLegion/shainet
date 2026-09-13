require "./spec_helper"

# The block chain's claim is a TRANSFER BUDGET at the stack level: a stack of N blocks
# costs ONE upload and ONE readback for the whole stack, not one per block, and the
# FFN's per-layer readback disappears entirely. Parity alone would not catch a
# regression that quietly restored those, so the counts are asserted directly.
private def block_ready?
  SHAInet::CUDA.fully_available? &&
    SHAInet::CUDA.block_device_kernels_available? &&
    SHAInet::CUDA.swiglu_kernel_available?
end

private def with_block_device(enabled : Bool?, &)
  prev = SHAInet::LlamaBlock.block_device_enabled?
  SHAInet::LlamaBlock.block_device_enabled = enabled
  begin
    yield
  ensure
    SHAInet::LlamaBlock.block_device_enabled = prev
  end
end

private def with_prof(&)
  prev = SHAInet::Profile.enabled?
  SHAInet::Profile.enabled = true
  SHAInet::Profile.reset
  begin
    yield
  ensure
    SHAInet::Profile.enabled = prev
    SHAInet::Profile.reset
  end
end

private def fill!(m, seed : Float64)
  m.rows.times do |i|
    m.cols.times { |j| m[i, j] = Math.sin(seed + i * 0.31 + j * 0.13) * 0.4 }
  end
  m
end

# A small MoE llama stack, quantized, so the device block path is genuinely available.
private def build_stack(layers : Int32, d_model : Int32, heads : Int32, kv_heads : Int32)
  blocks = Array(SHAInet::LlamaBlock).new
  layers.times do |li|
    b = SHAInet::LlamaBlock.new(d_model, heads, 128, num_kv_heads: kv_heads,
      moe_experts: 4, moe_top_k: 2, moe_ff_hidden: 128)
    fill!(b.w_q.as(SHAInet::SimpleMatrix), 1.0 + li)
    fill!(b.w_k.as(SHAInet::SimpleMatrix), 2.0 + li)
    fill!(b.w_v.as(SHAInet::SimpleMatrix), 3.0 + li)
    fill!(b.w_o.as(SHAInet::SimpleMatrix), 4.0 + li)
    ffn = b.ffn.as(SHAInet::MoEFF)
    fill!(ffn.router.as(SHAInet::SimpleMatrix), 5.0 + li)
    ffn.experts.each_with_index do |e, ei|
      e.gate_proj = fill!(SHAInet::SimpleMatrix.new(d_model, 128), 6.0 + li + ei)
      e.up_proj = fill!(SHAInet::SimpleMatrix.new(d_model, 128), 7.0 + li + ei)
      e.down_proj = fill!(SHAInet::SimpleMatrix.new(128, d_model), 8.0 + li + ei)
    end
    blocks << b
  end
  blocks
end

# Built ONCE and shared. Each example used to build and quantize its own 3-block stack,
# and nothing frees a quantized block's device memory, so the churn pushed the suite
# over a threshold where the pre-existing CudaMatrix GC-finalizer crash (a CUDA free
# running from a libgc finalizer) became reliable rather than rare. The two new spec
# files were each clean alone and crashed 3 of 4 runs together, which is what a
# cumulative allocation threshold looks like. Sharing the stack keeps the footprint
# flat; the underlying hazard is pre-existing and still worth fixing in src.
private STACK_HOLDER = [] of Array(SHAInet::LlamaBlock)

private def shared_stack
  if STACK_HOLDER.empty?
    s = build_stack(3, 64, 4, 2)
    s.each(&.to_gpu!(quantize: true, bits: 4))
    STACK_HOLDER << s
  end
  stack = STACK_HOLDER.first
  stack.each(&.clear_cache!)
  stack
end

describe "device-resident block chain" do
  it "matches the host path for a single decode step" do
    pending! "CUDA with the block kernels not available" unless block_ready?

    d_model = 64
    blocks = shared_stack
    blocks.each(&.block_device_capable?.should(be_true))

    x = fill!(SHAInet::SimpleMatrix.new(1, d_model), 0.2)

    host = x
    with_block_device(false) do
      blocks.each { |b| host = b.forward_cached(host) }
    end

    blocks.each(&.clear_cache!)
    dev_in = SHAInet::CudaMatrix.new(1, d_model)
    d_model.times { |c| dev_in[0, c] = x[0, c] }
    dev_in.mark_host_modified!
    dev_in.sync_to_device!("spec_in")
    cur = dev_in
    with_block_device(true) do
      blocks.each { |b| cur = b.forward_cached_device(cur) }
    end
    cur.sync_from_device!("spec_out") if cur.device_dirty?

    # 1e-2: the device norm accumulates its sum of squares in float where the host uses
    # Float64, and the difference compounds through three blocks. The binding
    # correctness check is the token-sequence equality measured on real models.
    d_model.times { |c| cur[0, c].should be_close(host[0, c], 1e-2) }
  ensure
    dev_in.try(&.free!)
  end

  it "spends one readback for the whole stack, not one per block" do
    pending! "CUDA with the block kernels not available" unless block_ready?

    d_model = 64
    layers = 3
    blocks = shared_stack

    dev_in = SHAInet::CudaMatrix.new(1, d_model)
    fill!(dev_in, 0.2)
    dev_in.mark_host_modified!
    dev_in.sync_to_device!("spec_in")

    with_prof do
      cur = dev_in
      with_block_device(true) do
        blocks.each { |b| cur = b.forward_cached_device(cur) }
      end
      stats = SHAInet::Profile.stats

      # Proves the device path ran rather than silently falling back.
      stats["block.dev_norm"].not_nil![:count].should eq layers * 2
      stats["block.dev_residual"].not_nil![:count].should eq layers * 2
      # Attention now runs on the device too: RoPE, QK-norm and the KV append are
      # kernels, so its projections never come home either.
      stats["attn.dev_qkv"].not_nil![:count].should eq layers
      stats["attn.dev_append"].not_nil![:count].should eq layers
      stats["attn.dev_oproj"].not_nil![:count].should eq layers

      # The FFN's per-layer readback is gone entirely.
      stats["ffn.dev_readback"]?.should be_nil

      # And so is EVERY per-matmul readback: gemm.out_d2h is the phase every
      # quantized matmul used to end with, and on this path it never fires. That is
      # the whole claim of the change, so it is asserted as an absence rather than a
      # smaller number.
      stats["gemm.out_d2h"]?.should be_nil
      stats["gemm.in_h2d"]?.should be_nil
    end
  ensure
    dev_in.try(&.free!)
  end

  it "the host path really does pay a readback per layer, with the same weights" do
    pending! "CUDA with the block kernels not available" unless block_ready?

    d_model = 64
    layers = 3
    blocks = shared_stack
    x = fill!(SHAInet::SimpleMatrix.new(1, d_model), 0.2)

    with_prof do
      cur = x
      # "The host path" now means BOTH device paths off. Device-resident prefill attention
      # also removes these readbacks, so with only the block chain disabled this example
      # would measure a path that is still half on the device and its counter-baseline
      # would be wrong.
      prev_prefill = SHAInet::LlamaBlock.prefill_attn_device_enabled?
      SHAInet::LlamaBlock.prefill_attn_device_enabled = false
      begin
        with_block_device(false) do
          blocks.each { |b| cur = b.forward_cached(cur) }
        end
      ensure
        SHAInet::LlamaBlock.prefill_attn_device_enabled = prev_prefill
      end
      stats = SHAInet::Profile.stats
      # The guard in the other direction: on the host path the FFN reads back once per
      # layer and the projections four times, so the budget above is a real reduction
      # and not an artifact of how the phases are named.
      stats["ffn.dev_readback"].not_nil![:count].should eq layers
      stats["gemm.out_d2h"].not_nil![:count].should eq layers * 4
    end
  end

  it "declines the device path when the block is not quantized" do
    d_model = 32
    blocks = build_stack(1, d_model, 4, 2)
    # No to_gpu!, so weights are SimpleMatrix and the device chain must refuse.
    blocks.first.block_device_capable?.should be_false
  end

  it "declines the device path when SHAINET_BLOCK_DEVICE is off" do
    pending! "CUDA with the block kernels not available" unless block_ready?

    blocks = shared_stack
    with_block_device(true) { blocks.first.block_device_capable?.should be_true }
    with_block_device(false) { blocks.first.block_device_capable?.should be_false }
  end
end
