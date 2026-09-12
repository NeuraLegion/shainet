require "./spec_helper"

# The super-block scale layout makes two claims and this spec asserts both
# directly, because either one alone can be met by a change that is useless:
#
#   1. SIZE: 4.375 bits per weight instead of 5.000 (12.5% smaller).
#   2. QUALITY: reconstruction error no materially worse than the fp32-per-32
#      layout it replaced.
#
# Both the OLD packer (as a baseline) and the dequantizers here are INDEPENDENT
# reimplementations of the documented layouts, not calls back into the packer, so
# a mis-indexed sub-scale or a wrong super-block stride fails rather than
# cancelling out.

private BLK   = 32
private SUPER =  8

# The layout this change replaced: one fp32 scale per 32-weight block. Kept here
# as the quality baseline; src has only the super-block packer now.
private def pack_flat(w : SHAInet::SimpleMatrix) : Tuple(Array(UInt8), Array(Float32))
  k = w.rows
  n = w.cols
  blocks = (k + BLK - 1) // BLK
  kbytes = (k + 1) // 2
  q = Array(UInt8).new(n * kbytes, 0_u8)
  s = Array(Float32).new(n * blocks, 0.0_f32)
  n.times do |col|
    blocks.times do |b|
      base = b * BLK
      lim = Math.min(base + BLK, k)
      max_abs = 0.0_f32
      (base...lim).each { |kk| v = w[kk, col].abs.to_f32; max_abs = v if v > max_abs }
      scale = max_abs > 0.0_f32 ? (max_abs / 7.0_f32).to_f32 : 1.0_f32
      s[col * blocks + b] = scale
      inv = 1.0_f32 / scale
      (base...lim).each do |kk|
        qv = (w[kk, col].to_f32 * inv).round
        qv = 7.0_f32 if qv > 7.0_f32
        qv = -7.0_f32 if qv < -7.0_f32
        nib = (qv.to_i + 8) & 0x0F
        idx = col * kbytes + (kk >> 1)
        if (kk & 1) == 0
          q[idx] = (q[idx] & 0xF0_u8) | nib.to_u8
        else
          q[idx] = (q[idx] & 0x0F_u8) | (nib.to_u8 << 4)
        end
      end
    end
  end
  {q, s}
end

private def dequant_flat(k : Int32, n : Int32, q : Array(UInt8), s : Array(Float32)) : Array(Float32)
  blocks = (k + BLK - 1) // BLK
  kbytes = (k + 1) // 2
  out = Array(Float32).new(k * n, 0.0_f32)
  n.times do |col|
    k.times do |kk|
      byte = q[col * kbytes + (kk >> 1)]
      nib = (kk & 1) == 1 ? (byte >> 4) : (byte & 0x0F)
      out[kk * n + col] = (nib.to_i - 8).to_f32 * s[col * blocks + (kk >> 5)]
    end
  end
  out
end

# Effective scale = d[block / SUPER] * sub[block] / 255.
private def dequant_super(k : Int32, n : Int32, q : Array(UInt8), d : Array(Float32), sub : Array(UInt8)) : Array(Float32)
  blocks = (k + BLK - 1) // BLK
  supers = (blocks + SUPER - 1) // SUPER
  kbytes = (k + 1) // 2
  out = Array(Float32).new(k * n, 0.0_f32)
  n.times do |col|
    k.times do |kk|
      b = kk >> 5
      byte = q[col * kbytes + (kk >> 1)]
      nib = (kk & 1) == 1 ? (byte >> 4) : (byte & 0x0F)
      eff = d[col * supers + (b // SUPER)] * sub[col * blocks + b].to_f32 / 255.0_f32
      out[kk * n + col] = (nib.to_i - 8).to_f32 * eff
    end
  end
  out
end

private def flat_bytes_for(k : Int32, n : Int32) : UInt64
  blocks = (k + BLK - 1) // BLK
  n.to_u64 * ((k + 1) // 2).to_u64 + n.to_u64 * blocks.to_u64 * 4_u64
end

private def rms_error(a : Array(Float32), ref : SHAInet::SimpleMatrix) : Float64
  sum = 0.0
  ref.rows.times do |i|
    ref.cols.times do |j|
      diff = a[i * ref.cols + j].to_f - ref[i, j].to_f
      sum += diff * diff
    end
  end
  Math.sqrt(sum / (ref.rows * ref.cols))
end

# A weight matrix with realistic structure: mostly small values with a few
# outliers, and DIFFERENT dynamic range per block, which is the case that
# stresses a shared super-block scale (a block whose scale is far below its
# super-block's maximum gets a coarser byte sub-scale).
private def spiky_weights(k : Int32, n : Int32) : SHAInet::SimpleMatrix
  m = SHAInet::SimpleMatrix.new(k, n)
  rng = Random.new(20260912)
  k.times do |i|
    n.times do |j|
      base = rng.next_float * 0.05 - 0.025
      base *= 50.0 if (i // BLK) % 5 == 0 # mixed scales inside a super-block
      base += (rng.next_float > 0.999 ? 1.5 : 0.0)
      m[i, j] = base
    end
  end
  m
end

describe "Q4 super-block scales" do
  it "is 4.375 bits per weight instead of 5.000" do
    k = 512
    n = 64
    weights = k * n

    flat = flat_bytes_for(k, n)
    sup = SHAInet::Q4CudaMatrix.device_bytes_for(k, n)

    # Assert the ABSOLUTE bits/weight of both, not just that one is smaller: a
    # ratio alone would still pass if both layouts silently got bigger.
    (flat.to_f * 8 / weights).should be_close(5.000, 1e-6)
    (sup.to_f * 8 / weights).should be_close(4.375, 1e-6)
    (1.0 - (sup.to_f / flat.to_f)).should be_close(0.125, 1e-6)
  end

  it "accounts for a partial super-block without under-allocating" do
    # K = 320 is 10 blocks = 1 full super-block + a partial one. Under-counting the
    # tail would be a device buffer overrun, so the accounting must round up.
    k = 320
    n = 8
    blocks = (k + BLK - 1) // BLK
    supers = (blocks + SUPER - 1) // SUPER
    blocks.should eq(10)
    supers.should eq(2)

    expected = n.to_u64 * ((k + 1) // 2).to_u64 + n.to_u64 * supers.to_u64 * 4_u64 + n.to_u64 * blocks.to_u64
    SHAInet::Q4CudaMatrix.device_bytes_for(k, n).should eq(expected)

    # And the packer must fill exactly those buffers.
    q, d, sub = SHAInet::Q4CudaMatrix.pack(spiky_weights(k, n))
    q.size.should eq(n * ((k + 1) // 2))
    d.size.should eq(n * supers)
    sub.size.should eq(n * blocks)
  end

  it "reconstructs no worse than the fp32-per-block layout" do
    k = 512
    n = 32
    w = spiky_weights(k, n)

    q_f, s_f = pack_flat(w)
    q_s, d_s, sub_s = SHAInet::Q4CudaMatrix.pack(w)

    err_flat = rms_error(dequant_flat(k, n, q_f, s_f), w)
    err_super = rms_error(dequant_super(k, n, q_s, d_s, sub_s), w)

    # Anchor the baseline against the SIGNAL rather than a magic constant, so this
    # stays meaningful if the fixture's magnitudes change. A change that wrecked
    # BOTH layouts fails here rather than passing on a ratio.
    w_rms = Math.sqrt(w.data.sum { |v| v.to_f * v.to_f } / (k * n))
    w_rms.should be > 0.0
    (err_flat / w_rms).should be < 0.10

    # Equal-quality claim: the byte sub-scale must not cost more than a few
    # percent of reconstruction error over the fp32 scale it replaced.
    (err_super / err_flat).should be < 1.05
  end

  it "never emits a zero sub-scale, which would make a block unrepresentable" do
    # A block whose values are tiny next to its super-block's outliers rounds
    # toward zero; clamping to 1 is what keeps it representable at all.
    _, d, sub = SHAInet::Q4CudaMatrix.pack(spiky_weights(512, 16))
    sub.each(&.should(be >= 1_u8))
    d.each(&.should(be > 0.0_f32))
  end

  it "packs an all-zero weight without dividing by zero" do
    q, d, sub = SHAInet::Q4CudaMatrix.pack(SHAInet::SimpleMatrix.new(64, 4))
    d.each(&.should(be > 0.0_f32))
    dequant_super(64, 4, q, d, sub).each(&.should(eq(0.0_f32)))
  end

  it "matches the CPU reference through the real GPU kernel" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    # K % 32 == 0 so this exercises the VECTORIZED kernel, which decodes the
    # super-block scales with its own indexing (kb / Q4_SUPER) separate from the
    # scalar path. Compared against the independent CPU dequant, not against fp32.
    k = 256
    n = 64
    w = spiky_weights(k, n)
    qm = SHAInet::Q4CudaMatrix.from_simple(w)
    q_s, d_s, sub_s = SHAInet::Q4CudaMatrix.pack(w)
    deq = dequant_super(k, n, q_s, d_s, sub_s)

    x = SHAInet::CudaMatrix.new(1, k)
    k.times { |j| x[0, j] = Math.sin(j * 0.13) * 0.5 }
    x.mark_host_modified!
    x.sync_to_device!("q4_super_spec_in")

    got = qm.gemv(x)
    got.sync_from_device!("q4_super_spec_out") if got.device_dirty?

    n.times do |col|
      want = 0.0
      k.times { |kk| want += x[0, kk].to_f * deq[kk * n + col].to_f }
      got[0, col].should be_close(want, 1e-3)
    end
  end

  it "matches the CPU reference through the scalar kernel too" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    # SHAINET_Q4_SCALAR forces the fallback kernel, which has its own scale decode.
    prev = ENV["SHAINET_Q4_SCALAR"]?
    ENV["SHAINET_Q4_SCALAR"] = "1"
    begin
      k = 256
      n = 32
      w = spiky_weights(k, n)
      qm = SHAInet::Q4CudaMatrix.from_simple(w)
      q_s, d_s, sub_s = SHAInet::Q4CudaMatrix.pack(w)
      deq = dequant_super(k, n, q_s, d_s, sub_s)

      x = SHAInet::CudaMatrix.new(1, k)
      k.times { |j| x[0, j] = Math.cos(j * 0.17) * 0.4 }
      x.mark_host_modified!
      x.sync_to_device!("q4_super_scalar_in")

      got = qm.gemv(x)
      got.sync_from_device!("q4_super_scalar_out") if got.device_dirty?

      n.times do |col|
        want = 0.0
        k.times { |kk| want += x[0, kk].to_f * deq[kk * n + col].to_f }
        got[0, col].should be_close(want, 1e-3)
      end
    ensure
      if prev
        ENV["SHAINET_Q4_SCALAR"] = prev
      else
        ENV.delete("SHAINET_Q4_SCALAR")
      end
    end
  end
end
