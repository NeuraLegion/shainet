require "./spec_helper"

# Q5_K support. Some i-quant GGUFs keep only the LM head as Q5_K (the JonathanColetti
# Qwen3.8-27B-Uncensored IQ2_M build stores output.weight that way), and before this type existed
# the model loaded, prefilled every layer, then died at the LM head with "unsupported GGUF type".
#
# Every path is checked against a direct transliteration of llama.cpp's dequantize_row_q5_K
# (ggml-quants.c), written the way ggml writes it -- walking pointers and shifting u1/u2 -- so it
# shares no structure with the kernels under test.

private def f16_bits(v : Float32) : UInt16
  # Only exact small values are used below, so a direct construction is enough: sign 0, and the
  # exponent/mantissa of a value that is a power of two times a short fraction.
  case v
  when  1.0_f32 then 0x3C00_u16
  when  0.5_f32 then 0x3800_u16
  when 0.25_f32 then 0x3400_u16
  else               raise "unsupported constant #{v}"
  end
end

private def put_f16(into : Pointer(UInt8), at : Int32, v : Float32)
  b = f16_bits(v)
  into[at] = (b & 0xFF).to_u8
  into[at + 1] = (b >> 8).to_u8
end

private def half_to_f32(h : UInt16) : Float32
  sign = (h >> 15) & 1
  exp = (h >> 10) & 0x1F
  man = h & 0x3FF
  v = if exp == 0
        man.to_f32 * 2.0_f32 ** -24
      elsif exp == 31
        Float32::INFINITY
      else
        (1.0_f32 + man.to_f32 / 1024.0_f32) * 2.0_f32 ** (exp.to_i - 15)
      end
  sign == 1 ? -v : v
end

private def ggml_scale_min(j : Int32, q : Pointer(UInt8)) : {Int32, Int32}
  if j < 4
    {(q[j] & 63).to_i, (q[j + 4] & 63).to_i}
  else
    {((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)).to_i, ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)).to_i}
  end
end

# llama.cpp dequantize_row_q5_K, transliterated.
private def ggml_dequant_q5k(x : Pointer(UInt8), k : Int32) : Array(Float32)
  y = [] of Float32
  (k // 256).times do |i|
    blk = x + i * 176
    d = half_to_f32(blk.as(Pointer(UInt16))[0])
    min = half_to_f32((blk + 2).as(Pointer(UInt16))[0])
    scales = blk + 4
    qh = blk + 16
    ql = blk + 48
    is = 0
    u1 = 1
    u2 = 2
    j = 0
    while j < 256
      sc, m = ggml_scale_min(is + 0, scales)
      d1 = d * sc; m1 = min * m
      sc, m = ggml_scale_min(is + 1, scales)
      d2 = d * sc; m2 = min * m
      32.times { |l| y << d1 * ((ql[l] & 0xF).to_i + (qh[l] & u1 != 0 ? 16 : 0)) - m1 }
      32.times { |l| y << d2 * ((ql[l] >> 4).to_i + (qh[l] & u2 != 0 ? 16 : 0)) - m2 }
      ql += 32; is += 2
      u1 <<= 2; u2 <<= 2
      j += 64
    end
  end
  y
end

# N rows of K values. Scales use all 6 bits including the high pair packed into bytes 0..7, and
# every qh bit pattern occurs, so a wrong scale unpack, nibble order or qh bit index cannot pass.
private def build_q5k(n : Int32, k : Int32) : Pointer(UInt8)
  nb = k // 256
  w = Pointer(UInt8).malloc(n * nb * 176)
  n.times do |r|
    nb.times do |b|
      blk = w + (r * nb + b) * 176
      put_f16(blk, 0, {1.0_f32, 0.5_f32, 0.25_f32}[(r + b) % 3])
      put_f16(blk, 2, {0.25_f32, 0.5_f32}[(r + b) % 2])
      12.times { |i| blk[4 + i] = ((i * 37 + r * 13 + b * 7 + 11) % 256).to_u8 }
      32.times { |i| blk[16 + i] = ((i * 53 + r * 29 + b * 3 + 5) % 256).to_u8 }
      128.times { |i| blk[48 + i] = ((i * 91 + r * 17 + b * 41 + 1) % 256).to_u8 }
    end
  end
  w
end

describe "Q5_K" do
  it "declares ggml's block size" do
    SHAInet::GGUF::BLOCK_SIZE[SHAInet::GGUF::GGMLType::Q5_K].should eq({176, 256})
  end

  it "host reference dequant matches llama.cpp's dequantize_row_q5_K" do
    pending! "CPU kernels not loaded" unless SHAInet::CPUKernels.available?
    k = 512
    n = 6
    w = build_q5k(n, k)
    buf = Pointer(Float32).malloc(k)
    distinct = Set(Float32).new
    n.times do |r|
      row = w + r * (k // 256) * 176
      want = ggml_dequant_q5k(row, k)
      SHAInet::CPUKernels.dequant_row(SHAInet::GGUF::GGMLType::Q5_K.value.to_i32, row, buf, k).should be_true
      k.times do |i|
        buf[i].should eq want[i]
        distinct << want[i]
      end
    end
    # Vacuity: a builder that produced a constant block would make any decoder agree.
    distinct.size.should be > 200
  end

  it "device GEMV (M=1) and prefill GEMM (M>1) match the reference", tags: "cuda" do
    pending! "no CUDA" unless SHAInet::CUDA.fully_available?
    pending! "no dequant kernels" unless SHAInet::CUDA.dequant_k_rows_available?
    SHAInet::GGUFMatrix.device_type_supported?(SHAInet::GGUF::GGMLType::Q5_K).should be_true

    k = 512
    n = 9 # not a multiple of the 4 warps per block, so the edge block runs
    w_host = build_q5k(n, k)
    refs = (0...n).map { |r| ggml_dequant_q5k(w_host + r * (k // 256) * 176, k) }
    w = SHAInet::GGUFMatrix.new(k, n, SHAInet::GGUF::GGMLType::Q5_K, w_host, (n * (k // 256) * 176).to_u64)

    [1, 5].each do |m|
      x = SHAInet::CudaMatrix.new(m, k)
      # Sign-varied so a min/offset error cannot cancel against an all-positive activation.
      (m * k).times { |i| x.raw_data[i] = (((i * 11) % 17) - 8) * 0.03_f32 }
      x.mark_host_modified!
      x.sync_to_device!("q5k_x")

      y = SHAInet::CudaMatrix.new(m, n)
      w.gemv_into(x, y) # M=1 -> GEMV kernel, M>1 -> dequant rows + cuBLAS
      SHAInet::CUDA.device_synchronize
      y.mark_device_dirty!
      y.sync_from_device!("q5k_y")

      m.times do |t|
        n.times do |c|
          want = 0.0
          mag = 0.0
          k.times do |i|
            p = refs[c][i].to_f * x.raw_data[t * k + i].to_f
            want += p
            mag += p.abs
          end
          got = y.raw_data[t * n + c].to_f
          # Error is measured against the sum of |terms|, not the result: the synthetic weights reach
          # ~2000 and the signed activation cancels much of the sum, so a result-relative bound would
          # be judging rounding noise. M=1 is an fp32 GEMV (tight); M>1 is cuBLAS in TF32, whose
          # 10-bit mantissa gives ~5e-4 per product.
          tol = m == 1 ? 1e-5 : 2e-3
          ((got - want).abs / mag).should be < tol
        end
      end
      x.free!
      y.free!
    end
    w.free!
    SHAInet::GGUFMatrix.release_scratch!
  end
end
