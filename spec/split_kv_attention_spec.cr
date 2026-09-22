require "./spec_helper"

# Split-KV attention parallelizes the decode softmax over the KV length and merges per-split partials
# with a max-rescale. That is exact in theory; this proves it against the kernel it replaces, which
# computes the whole row in one block. A softmax rewrite is exactly the kind of change that stays
# plausible-looking while being subtly wrong, so the comparison spans several context lengths --
# including ones that straddle the 512-key split boundary, where an off-by-one in the range or a
# mishandled empty split would show up.
describe "split-KV decode attention" do
  it "matches the single-block kernel across context lengths" do
    unless SHAInet::CUDA.fully_available? && SHAInet::CUDA.attention_split_kv_available? &&
           SHAInet::CUDA.attention_device_kernels_available?
      pending! "CUDA attention kernels unavailable"
    end

    num_heads = 4
    heads_per_kv = 2
    kv_heads = num_heads // heads_per_kv
    head_dim = 32
    new_tokens = 1
    scale = (1.0 / Math.sqrt(head_dim.to_f)).to_f32

    # 1 exercises the degenerate single-key row; 511/512/513 straddle the split boundary; 1100 needs
    # three splits with the last one partly filled.
    [1, 511, 512, 513, 1100].each do |ctx|
      start_pos = ctx - new_tokens
      capacity = ctx + 8

      q = SHAInet::CudaMatrix.new(new_tokens, num_heads * head_dim)
      (new_tokens * num_heads * head_dim).times { |i| q.raw_data[i] = ((i * 13 % 17) - 8) * 0.11_f32 }
      q.mark_host_modified!
      q.sync_to_device!("q")

      kc = SHAInet::CudaMatrix.new(kv_heads, capacity * head_dim)
      vc = SHAInet::CudaMatrix.new(kv_heads, capacity * head_dim)
      (kv_heads * capacity * head_dim).times do |i|
        kc.raw_data[i] = ((i * 7 % 19) - 9) * 0.07_f32
        vc.raw_data[i] = ((i * 11 % 23) - 11) * 0.05_f32
      end
      kc.mark_host_modified!; kc.sync_to_device!("kc")
      vc.mark_host_modified!; vc.sync_to_device!("vc")

      split_out = SHAInet::CudaMatrix.new(new_tokens, num_heads * head_dim)
      plain_out = SHAInet::CudaMatrix.new(new_tokens, num_heads * head_dim)

      ws_floats = SHAInet::CUDA.attention_split_ws_floats(new_tokens, num_heads, head_dim, ctx)
      ws_floats.should be > 0
      split_ws = SHAInet::CudaMatrix.new(1, ws_floats)
      # The single-block kernel stages the whole score row in its workspace.
      plain_ws = SHAInet::CudaMatrix.new(num_heads * new_tokens, ctx)

      ok = SHAInet::CUDA.attention_split_kv_f32(
        q.device_ptr.not_nil!, kc.device_ptr.not_nil!, vc.device_ptr.not_nil!,
        split_out.device_ptr.not_nil!, split_ws.device_ptr.not_nil!,
        new_tokens, start_pos, num_heads, heads_per_kv, head_dim, capacity, scale)
      ok.should be_true

      SHAInet::CUDA.attention_kv_f32(
        q.device_ptr.not_nil!, kc.device_ptr.not_nil!, vc.device_ptr.not_nil!,
        plain_out.device_ptr.not_nil!, plain_ws.device_ptr.not_nil!,
        new_tokens, start_pos, num_heads, heads_per_kv, head_dim, capacity, scale)
      SHAInet::CUDA.device_synchronize

      split_out.mark_device_dirty!; split_out.sync_from_device!("split")
      plain_out.mark_device_dirty!; plain_out.sync_from_device!("plain")

      n = new_tokens * num_heads * head_dim
      # Both paths exponentiate and sum in a different order, so allow accumulation slack but not
      # enough to hide a wrong weighting.
      n.times do |i|
        split_out.raw_data[i].should be_close(plain_out.raw_data[i], 2e-5)
      end

      # Vacuity: a softmax-weighted average of non-zero V must not be all zeros, which is what a
      # kernel that silently wrote nothing would produce and the comparison above would accept if the
      # reference were broken the same way.
      nonzero = n.times.count { |i| split_out.raw_data[i].abs > 1e-6 }
      nonzero.should be > 0

      [q, kc, vc, split_out, plain_out, split_ws, plain_ws].each(&.free!)
    end
  end
end
