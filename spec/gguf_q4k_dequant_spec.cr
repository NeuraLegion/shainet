require "./spec_helper"

# Correctness spec for the Q4_K dequantization path behind CUDA.gemv_q4k.
#
# Instead of comparing against fp32 (which only bounds quantization error), we
# build ONE Q4_K block (144 bytes) by hand with known scales/mins/nibbles, feed
# it an all-ones activation vector, and assert the GEMV reduces to the exact
# dequant sum. A wrong nibble, a mis-unpacked 6-bit scale, or a dropped
# group would land far outside the fp16 tolerance.
#
# Q4_K block layout (144 bytes, K = 256 per block):
#   [0..1]    d     (fp16)   -- super-block scale
#   [2..3]    dmin  (fp16)   -- super-block min
#   [4..15]   scales[12]     -- 8 group scales + 8 group mins packed 6-bit
#   [16..143] qs[128]        -- 256 4-bit quants (low nibble first)
#
# dequant(low nibble)  = d * group_scale * q_lo - dmin * group_min
# dequant(high nibble) = d * group_scale * q_hi - dmin * group_min

# Little-endian fp16 encoder for the two constants we need. 1.0 = 0x3C00,
# 0.5 = 0x3800 in IEEE half.
private def fp16_bytes(bits : UInt16) : Tuple(UInt8, UInt8)
  {(bits & 0x00FF).to_u8, ((bits >> 8) & 0x00FF).to_u8}
end

# Build the 144-byte Q4_K block described in the spec:
#   d = 1.0, dmin = 0.5
#   all 8 group scales = 1, all 8 group mins = 1
#   all 128 qs bytes = 0x53 (low nibble 3, high nibble 5)
private def build_q4k_block : Bytes
  block = Bytes.new(144, 0_u8)

  # d = 1.0 (fp16 0x3C00), dmin = 0.5 (fp16 0x3800), little-endian.
  d_lo, d_hi = fp16_bytes(0x3C00_u16)
  block[0] = d_lo
  block[1] = d_hi
  dmin_lo, dmin_hi = fp16_bytes(0x3800_u16)
  block[2] = dmin_lo
  block[3] = dmin_hi

  # scales[12] via the get_scale_min_k4 inverse for all scales=1, mins=1:
  #   groups 0-3: scales[0..3] = 1, scales[4..7] = 1
  #   groups 4-7: scales[8..11] = (1 & 0xF) | ((1 & 0xF) << 4) = 0x11
  #               high 2 bits of scale/min are 0 (1 >> 4 == 0), so no OR-back.
  scales = block[4, 12]
  4.times do |j|
    scales[j] = 1_u8        # group j scale (low 6 bits)
    scales[j + 4] = 1_u8    # group j min   (low 6 bits)
    scales[j + 8] = 0x11_u8 # group j+4: packed 4-bit scale/min
  end

  # qs[128] all 0x53: low nibble 3, high nibble 5.
  128.times { |i| block[16 + i] = 0x53_u8 }

  block
end

describe "Q4_K dequantization" do
  it "reduces an all-ones GEMV to the exact dequant sum", tags: "cuda" do
    pending! "no CUDA" unless SHAInet::CUDA.fully_available?

    block = build_q4k_block

    # Upload the raw block to device memory.
    w_ptr = Pointer(Void).null
    SHAInet::CUDA.malloc(pointerof(w_ptr), LibC::SizeT.new(block.size))
    SHAInet::CUDA.memcpy(w_ptr, block.to_unsafe.as(Pointer(Void)),
      LibC::SizeT.new(block.size), SHAInet::CUDA::MemcpyKind::HostToDevice)

    # 1x256 input of all 1.0s on device.
    x_host = Array(Float32).new(256, 1.0_f32)
    x_ptr = Pointer(Void).null
    x_bytes = LibC::SizeT.new(256 * sizeof(Float32))
    SHAInet::CUDA.malloc(pointerof(x_ptr), x_bytes)
    SHAInet::CUDA.memcpy(x_ptr, x_host.to_unsafe.as(Pointer(Void)),
      x_bytes, SHAInet::CUDA::MemcpyKind::HostToDevice)

    # 1x1 output on device.
    y_ptr = Pointer(Void).null
    y_bytes = LibC::SizeT.new(sizeof(Float32))
    SHAInet::CUDA.malloc(pointerof(y_ptr), y_bytes)

    # y[1,1] = x[1,256] * dequant(W[1,256]) ; M=1, N=1, K=256.
    SHAInet::CUDA.gemv_q4k(
      x_ptr.as(Pointer(Float32)),
      w_ptr.as(Pointer(UInt8)),
      y_ptr.as(Pointer(Float32)),
      1, 1, 256)
    SHAInet::CUDA.device_synchronize

    # Read the result back.
    y_host = Array(Float32).new(1, 0.0_f32)
    SHAInet::CUDA.memcpy(y_host.to_unsafe.as(Pointer(Void)), y_ptr,
      y_bytes, SHAInet::CUDA::MemcpyKind::DeviceToHost)

    SHAInet::CUDA.free(x_ptr)
    SHAInet::CUDA.free(y_ptr)
    SHAInet::CUDA.free(w_ptr)

    # 128 low nibbles dequant to 1.0*1*3 - 0.5*1 = 2.5,
    # 128 high nibbles dequant to 1.0*1*5 - 0.5*1 = 4.5.
    # Sum with an all-ones activation: 128*2.5 + 128*4.5 = 896.0.
    # 1.0 tolerance absorbs the fp16 scale rounding.
    y_host[0].to_f64.should be_close(896.0, 1.0)
  end
end
