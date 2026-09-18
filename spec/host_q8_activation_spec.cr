require "./spec_helper"

# The host k-quant path quantizes the ACTIVATION to int8 and multiplies against the raw weight
# bytes with integer SIMD, which is llama.cpp's vec_dot_q4_K_q8_K strategy and the reason Ollama
# stays fast with layers offloaded. It is a deliberate accuracy trade, so pin the size of it:
# llama.cpp ships 0.1-0.6% here, and measured on real Qwen3.8-27B weights this path lands at
# 0.283% rms with cosine >= 0.99998. A bug in the nibble unpacking, the 6-bit scale unpacking, or
# the Q4_K min term would land orders of magnitude outside that, not slightly outside.
#
# The reference is computed from the SAME bytes by straightforward fp32 dequantization, so the
# only difference under test is the int8 activation, not the weight interpretation.

private def f16(bits : UInt16, into : Bytes, at : Int32)
  into[at] = (bits & 0xFF).to_u8
  into[at + 1] = (bits >> 8).to_u8
end

# d = 1.0, dmin = 0.5, all 8 group scales and mins = 1, qs varying by index.
private def q4k_block(seed : Int32) : Bytes
  b = Bytes.new(144, 0_u8)
  f16(0x3C00_u16, b, 0)
  f16(0x3800_u16, b, 2)
  6.times { |i| b[4 + i] = 0x41_u8 }
  6.times { |i| b[10 + i] = 0x41_u8 }
  128.times { |i| b[16 + i] = (((i + seed) % 13) | ((((i + seed) + 5) % 11) << 4)).to_u8 }
  b
end

private def q6k_block(seed : Int32) : Bytes
  b = Bytes.new(210, 0_u8)
  128.times { |i| b[i] = (((i + seed) * 7) % 251).to_u8 }
  64.times { |i| b[128 + i] = (((i + seed) * 11) % 253).to_u8 }
  16.times { |i| b[192 + i] = (i.odd? ? -(i + 1) : (i + 1)).to_i8.to_u8! }
  f16(0x3C00_u16, b, 208)
  b
end

# fp32 reference: dequantize the block exactly as llama.cpp's dequantize_row_* does.
private def deq_q4k(b : Bytes) : Array(Float32)
  out = Array(Float32).new(256, 0.0_f32)
  d = 1.0_f32
  dmin = 0.5_f32
  scales = b[4, 12]
  qs = b[16, 128]
  4.times do |j64|
    2.times do |half|
      j = j64 * 2 + half
      sc, mn = if j < 4
                 {scales[j] & 63, scales[j + 4] & 63}
               else
                 {(scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
                  (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)}
               end
      32.times do |l|
        q = half == 0 ? (qs[j64 * 32 + l] & 0xF) : (qs[j64 * 32 + l] >> 4)
        out[j64 * 64 + half * 32 + l] = d * sc.to_f32 * q.to_f32 - dmin * mn.to_f32
      end
    end
  end
  out
end

private def deq_q6k(b : Bytes) : Array(Float32)
  out = Array(Float32).new(256, 0.0_f32)
  d = 1.0_f32
  2.times do |chunk|
    ql = b[chunk * 64, 64]
    qh = b[128 + chunk * 32, 32]
    sc = b[192 + chunk * 8, 8]
    32.times do |l|
      q = [
        ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)).to_i - 32,
        ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)).to_i - 32,
        ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)).to_i - 32,
        ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)).to_i - 32,
      ]
      4.times do |sub|
        s = sc[(l // 16) + sub * 2].to_i8!
        out[chunk * 128 + sub * 32 + l] = d * s.to_f32 * q[sub].to_f32
      end
    end
  end
  out
end

describe "host Q8 activation path" do
  {"Q4_K" => 144, "Q6_K" => 210}.each do |kind, bs|
    it "stays within the quantization trade for #{kind}" do
      pending! "CPU kernels not loaded" unless SHAInet::CPUKernels.available?

      k = 256
      n = 6
      m = 3
      w = Pointer(UInt8).malloc(n * bs)
      ref_rows = Array(Array(Float32)).new
      n.times do |r|
        blk = kind == "Q4_K" ? q4k_block(r) : q6k_block(r)
        blk.each_with_index { |byte, i| w[r * bs + i] = byte }
        ref_rows << (kind == "Q4_K" ? deq_q4k(blk) : deq_q6k(blk))
      end

      x = Pointer(Float32).malloc(m * k)
      (m * k).times { |i| x[i] = (0.05 * Math.sin(i * 0.37) + 0.01).to_f32 }

      y = Pointer(Float32).malloc(m * n)
      ok = if kind == "Q4_K"
             SHAInet::CPUKernels.gemv_q4k(x, w, y, m, n, k)
           else
             SHAInet::CPUKernels.gemv_q6k(x, w, y, m, n, k)
           end
      pending! "kernel unavailable" unless ok

      # Exact fp32 reference from the same bytes.
      num = 0.0
      den = 0.0
      m.times do |r|
        n.times do |c|
          exact = 0.0
          k.times { |i| exact += ref_rows[c][i].to_f64 * x[r * k + i].to_f64 }
          diff = y[r * n + c].to_f64 - exact
          num += diff * diff
          den += exact * exact
        end
      end
      rms_rel = Math.sqrt(num / den)
      # Comfortably above the ~0.3% measured on real weights, far below any unpacking error.
      rms_rel.should be < 0.02
    end
  end
end
