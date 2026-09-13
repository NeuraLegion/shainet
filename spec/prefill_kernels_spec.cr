require "./spec_helper"

# Parity for the multi-row prefill kernels against the host math they replace. These are
# the foundation the device-resident prefill attention is built on, so they are verified
# independently: a wrong RoPE or a wrong repack would surface later as plausible-looking
# but incorrect tokens, which is the hardest kind of bug to find.
private def kernels_ready?
  SHAInet::CUDA.fully_available? && SHAInet::CUDA.prefill_attn_kernels_available?
end

private def dev(m : SHAInet::SimpleMatrix) : SHAInet::CudaMatrix
  g = m.to_cuda
  g.sync_to_device!("spec_in") unless g.device_dirty?
  g
end

private def host_back(g : SHAInet::CudaMatrix) : SHAInet::SimpleMatrix
  g.sync_from_device!("spec_out") if g.device_dirty?
  g.to_simple
end

private def filled(rows, cols, seed = 0.0)
  m = SHAInet::SimpleMatrix.new(rows, cols)
  rows.times { |i| cols.times { |j| m[i, j] = Math.sin(seed + i * 0.37 + j * 0.19) * 0.6 } }
  m
end

describe "multi-row prefill kernels" do
  it "applies RoPE per row at consecutive positions" do
    pending! "CUDA/kernels not available" unless kernels_ready?

    rows = 5
    heads = 3
    head_dim = 8
    half = head_dim // 2
    base_pos = 7
    src = filled(rows, heads * head_dim, 0.2)

    # Host reference: the same half-split rotation the CPU prefill path does, each row at
    # base_pos + r.
    want = filled(rows, heads * head_dim, 0.2)
    inv = Array(Float32).new(half) { |i| (1.0 / (10000.0 ** (2.0 * i / head_dim))).to_f32 }
    rows.times do |r|
      pos = base_pos + r
      heads.times do |h|
        base = h * head_dim
        half.times do |i|
          angle = (pos * inv[i]).to_f32
          c = Math.cos(angle).to_f32
          s = Math.sin(angle).to_f32
          x0 = src[r, base + i].to_f32
          x1 = src[r, base + i + half].to_f32
          want[r, base + i] = x0 * c - x1 * s
          want[r, base + i + half] = x1 * c + x0 * s
        end
      end
    end

    g = dev(src)
    freq = SHAInet::SimpleMatrix.new(1, half)
    half.times { |i| freq[0, i] = inv[i] }
    fg = dev(freq)
    SHAInet::CUDA.rope_forward_rows(g.device_ptr.not_nil!, fg.device_ptr.not_nil!,
      base_pos, rows, heads, head_dim)
    g.mark_device_dirty!
    got = host_back(g)

    rows.times do |r|
      (heads * head_dim).times { |c| got[r, c].should be_close(want[r, c], 1e-4) }
    end
    g.free!
    fg.free!
  end

  it "normalizes each head of each row independently" do
    pending! "CUDA/kernels not available" unless kernels_ready?

    rows = 4
    heads = 3
    head_dim = 8
    eps = 1e-6
    src = filled(rows, heads * head_dim, 1.1)
    gamma = SHAInet::SimpleMatrix.new(1, head_dim)
    head_dim.times { |j| gamma[0, j] = 0.5 + j * 0.1 }

    want = SHAInet::SimpleMatrix.new(rows, heads * head_dim)
    rows.times do |r|
      heads.times do |h|
        base = h * head_dim
        sum = 0.0
        head_dim.times { |j| v = src[r, base + j].to_f; sum += v * v }
        inv = 1.0 / Math.sqrt(sum / head_dim + eps)
        head_dim.times { |j| want[r, base + j] = src[r, base + j].to_f * inv * gamma[0, j].to_f }
      end
    end

    g = dev(src)
    gg = dev(gamma)
    SHAInet::CUDA.head_rmsnorm_rows(g.device_ptr.not_nil!, gg.device_ptr.not_nil!,
      rows, heads, head_dim, eps.to_f32)
    g.mark_device_dirty!
    got = host_back(g)

    rows.times do |r|
      (heads * head_dim).times { |c| got[r, c].should be_close(want[r, c], 1e-4) }
    end
    g.free!
    gg.free!
  end

  it "broadcasts a bias over rows" do
    pending! "CUDA/kernels not available" unless kernels_ready?

    rows = 6
    cols = 10
    src = filled(rows, cols, 2.3)
    bias = SHAInet::SimpleMatrix.new(1, cols)
    cols.times { |j| bias[0, j] = j * 0.25 - 1.0 }

    g = dev(src)
    bg = dev(bias)
    SHAInet::CUDA.add_bias_rows(g.device_ptr.not_nil!, bg.device_ptr.not_nil!, rows, cols)
    g.mark_device_dirty!
    got = host_back(g)

    rows.times do |r|
      cols.times { |c| got[r, c].should be_close(src[r, c].to_f + bias[0, c].to_f, 1e-5) }
    end
    g.free!
    bg.free!
  end

  it "repacks token-major KV into kv-head-major" do
    pending! "CUDA/kernels not available" unless kernels_ready?

    rows = 5
    kv_heads = 3
    head_dim = 4
    src = filled(rows, kv_heads * head_dim, 3.7)

    # The layout the KV append expects: all of kv_head 0's tokens, then kv_head 1's.
    g = dev(src)
    dst = SHAInet::CudaMatrix.new(1, rows * kv_heads * head_dim)
    SHAInet::CUDA.pack_kv_heads(dst.device_ptr.not_nil!, g.device_ptr.not_nil!,
      rows, kv_heads, head_dim)
    dst.mark_device_dirty!
    got = host_back(dst)

    kv_heads.times do |kv_h|
      rows.times do |r|
        head_dim.times do |d|
          want = src[r, kv_h * head_dim + d].to_f
          idx = kv_h * rows * head_dim + r * head_dim + d
          got[0, idx].should be_close(want, 1e-6)
        end
      end
    end
    g.free!
    dst.free!
  end
end
