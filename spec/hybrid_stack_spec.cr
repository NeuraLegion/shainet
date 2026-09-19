require "./spec_helper"

# A hybrid stack is the point of all of this: three linear-attention layers for every one full
# attention layer, dispatched per layer from the config's layer_types list. These examples build
# one programmatically and inspect it, which is the last thing verifiable without a real Qwen3.5
# checkpoint on disk.
describe "hybrid layer stack" do
  it "builds a gated_deltanet layer as the linear-attention type" do
    net = SHAInet::Network.new
    net.add_layer("embedding", 16, vocab_size: 50)
    net.add_layer("gated_deltanet", 16, num_heads: 4, ff_hidden: 32, num_kv_heads: 2, head_dim: 4)

    # It must actually BE the linear-attention block, not a silently substituted default -- the
    # builder's else branch would hand back a plain MatrixLayer and everything would still run.
    net.hidden_layers.any?(SHAInet::GatedDeltaNetBlock).should be_true
  end

  it "interleaves gated_deltanet and llama blocks per layer_types" do
    net = SHAInet::Network.new
    net.add_layer("embedding", 16, vocab_size: 50)
    types = SHAInet::HFLoader.default_layer_types(4)
    types.each do |t|
      name = t == "linear_attention" ? "gated_deltanet" : "llama"
      net.add_layer(name, 16, num_heads: 4, ff_hidden: 32, num_kv_heads: 2, head_dim: 4)
    end

    linear = net.hidden_layers.count { |l| l.is_a?(SHAInet::GatedDeltaNetBlock) }
    full = net.hidden_layers.count { |l| l.is_a?(SHAInet::LlamaBlock) }
    linear.should eq(3)
    full.should eq(1)
  end

  it "honours the conv kernel from the config" do
    net = SHAInet::Network.new
    net.add_layer("gated_deltanet", 16, num_heads: 4, ff_hidden: 32, num_kv_heads: 2,
      head_dim: 4, linear_conv_kernel: 6)
    block = net.hidden_layers.find!(&.is_a?(SHAInet::GatedDeltaNetBlock))
    block.as(SHAInet::GatedDeltaNetBlock).conv_kernel.should eq(6)
  end

  # KNOWN LIMITATION, pinned so it cannot go quiet.
  #
  # @transformer_layers is typed Array(TransformerLayer | LlamaLayer) and read from 24 sites,
  # several on the hot inference path, so widening that union is still deferred. Network#clear_cache!
  # therefore reaches these blocks through @hidden_layers instead.
  #
  # This example previously asserted the OPPOSITE -- that a network-level clear did NOT reach the
  # block -- on the stated grounds that no loader could build a hybrid stack from a checkpoint. GGUF
  # support for Qwen3.5/3.8 made that false, and the gap became a real bug: recurrent state is not a
  # stale prefix that gets overwritten, it feeds the first token of the next sequence, so a second
  # prompt in the same process decoded an immediate end-of-turn and then empty im_start/im_end pairs
  # forever while the same prompt in a fresh process answered correctly.
  it "is reached by Network#clear_cache!, so a second sequence starts clean" do
    net = SHAInet::Network.new
    net.add_layer("gated_deltanet", 16, num_heads: 4, ff_hidden: 32, num_kv_heads: 2, head_dim: 4)
    block = net.hidden_layers.find!(&.is_a?(SHAInet::GatedDeltaNetBlock))
      .as(SHAInet::GatedDeltaNetBlock)

    # add_layer leaves the weights at zero, and with w_v zero NOTHING is ever written to the
    # recurrent state, so the output would be state-independent and this example would pass
    # vacuously. Give it real weights first.
    # The five large projections now hold a union (they can be quantized or device-resident), so
    # they are cast back to the host form a freshly built block still has.
    [block.w_q.as(SHAInet::SimpleMatrix), block.w_k.as(SHAInet::SimpleMatrix),
     block.w_v.as(SHAInet::SimpleMatrix), block.w_o.as(SHAInet::SimpleMatrix),
     block.w_gate.as(SHAInet::SimpleMatrix), block.w_alpha, block.w_beta,
     block.conv_q.weight, block.conv_k.weight, block.conv_v.weight].each_with_index do |m, idx|
      m.rows.times do |i|
        m.cols.times { |j| m[i, j] = 0.08 * Math.sin((i * 7 + j * 13 + idx * 3) * 0.37) }
      end
    end

    x = SHAInet::SimpleMatrix.new(4, 16, 0.0)
    4.times { |t| 16.times { |j| x[t, j] = 0.3 * Math.sin((t * 3 + j) * 0.2) } }

    first = block.forward(x)
    net.clear_cache!
    after_net_clear = block.forward(x)

    # The network-level clear reaches the block, so the second pass reproduces the first exactly.
    4.times { |t| 16.times { |j| after_net_clear[t, j].should be_close(first[t, j], 1e-6) } }

    # Counter-check, so the assertion above cannot pass vacuously on a state-independent block:
    # running again WITHOUT clearing must differ.
    dirty = block.forward(x)
    diff = 0.0
    4.times { |t| 16.times { |j| diff += (dirty[t, j] - first[t, j]).abs } }
    diff.should be > 1e-6
  end

  it "a linear layer's whole state is worth only a few hundred positions of one KV cache" do
    # The capacity argument in one assertion, at real Qwen3.5-ish dimensions.
    linear = SHAInet::GatedDeltaNetBlock.new(2048, 6144, 32, 16, 128, 128, 4)
    fixed = linear.state_bytes

    # 96 KiB per position is the measured fp16 KV cost per layer on the 30B.
    positions_to_match = fixed // (96 * 1024)
    positions_to_match.should be < 200_i64

    # And unlike a cache, it does not grow.
    x = SHAInet::SimpleMatrix.new(8, 2048, 0.01)
    linear.forward(x)
    linear.state_bytes.should eq(fixed)
  end
end
