require "./spec_helper"

# The i-quant kernels, each verified against the scalar reference ported from ggml-quants.c.
#
# Both halves of the check matter. The reference is a direct port, so agreeing with it proves the
# CUDA kernel decodes the format the same way the port does -- and the port itself is exercised on
# blocks taken from a REAL GSQ-RCO file rather than bytes I invented. That second part is not
# ceremony: a hand-built block tends to have small, similar scales and an easy sign pattern, so it
# can satisfy a kernel that misreads a split scale field or a packed sign selector. Real tensors
# spread across the whole range of both.
MODEL_IQ = "/home/unshadow/models/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"

# A DIFFERENTLY quantized build of the same checkpoint, used as ground truth for the reference
# dequants. Its norms match the GSQ-RCO file's at cosine 1.0, which is what establishes that the two
# files really are the same weights and makes the comparison below meaningful.
MODEL_TRUSTED = "/home/unshadow/.ollama/models/blobs/" \
                "sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d"

# type => {block bytes, reference dequant}
IQ_CASES = {
  SHAInet::GGUF::GGMLType::IQ4_XS  => 136,
  SHAInet::GGUF::GGMLType::IQ3_S   => 110,
  SHAInet::GGUF::GGMLType::IQ3_XXS => 98,
}

describe "i-quant kernels" do
  # Every reference dequant, checked against the SAME tensor in a differently quantized build.
  #
  # This is the check that matters most, and the one whose absence cost the most. The kernels were
  # verified against the references and the references against each other's shape, but only five of
  # the eight types were ever compared to independent ground truth -- and both unchecked types were
  # wrong. IQ2_XXS read its sub-blocks at a 4-byte stride where ggml advances a uint16_t* by 4 (8
  # bytes), giving cosine 0.117; IQ2_XS read its scales at offset 2+32, inside qs, instead of 2+64,
  # giving 0.780 -- high enough to pass for ordinary low-bit loss. Together they corrupted the
  # attention gate of every full-attention layer and the model produced a logit distribution with no
  # peak.
  #
  # So this enumerates the types PRESENT IN THE FILE rather than a list written by hand. A list is
  # exactly how the gap happened: a type nobody thought to add is a type nobody tests.
  it "dequantizes every type in the file to agree with a differently quantized build" do
    pending! "CPU kernels unavailable" unless SHAInet::CPUKernels.available?
    pending! "model not present" unless File.exists?(MODEL_IQ)
    pending! "trusted model not present" unless File.exists?(MODEL_TRUSTED)

    gq = SHAInet::GGUF::File.open(MODEL_IQ)
    gt = SHAInet::GGUF::File.open(MODEL_TRUSTED)

    # One representative tensor per type, restricted to names both files carry. Float types are
    # excluded: they are read directly rather than through a block dequant, so they have no
    # reference to check. This mirrors the loader's own quantized-vs-float split, which is a real
    # distinction -- BLOCK_SIZE registers F32/F16/BF16 with a block of one value, so "has a block
    # size" does not mean "is quantized".
    floats = {SHAInet::GGUF::GGMLType::F32, SHAInet::GGUF::GGMLType::F16,
              SHAInet::GGUF::GGMLType::BF16}
    per_type = {} of SHAInet::GGUF::GGMLType => String
    gq.tensors.each do |nm, info|
      next unless nm.starts_with?("blk.") && nm.ends_with?(".weight")
      next unless gt.tensors.has_key?(nm)
      next if floats.includes?(info.type) || floats.includes?(gt.tensors[nm].type)
      per_type[info.type] ||= nm
    end
    per_type.size.should be > 4

    per_type.each do |type, nm|
      iq = gq.tensors[nm]
      it_ = gt.tensors[nm]
      k = iq.shape[0].to_i32
      bq, vq = SHAInet::GGUF::BLOCK_SIZE[type]
      bt, vt = SHAInet::GGUF::BLOCK_SIZE[it_.type]
      rbq = (k // vq) * bq
      rbt = (k // vt) * bt
      pq = gq.tensor_ptr(iq).not_nil!
      pt = gt.tensor_ptr(it_).not_nil!
      bufq = Pointer(Float32).malloc(k)
      buft = Pointer(Float32).malloc(k)

      [0, 11, 97].each do |row|
        SHAInet::CPUKernels.dequant_row(type.value.to_i32, pq + row.to_i64 * rbq, bufq, k).should be_true
        SHAInet::CPUKernels.dequant_row(it_.type.value.to_i32, pt + row.to_i64 * rbt, buft, k).should be_true
        dot = 0.0
        nq = 0.0
        nt = 0.0
        k.times do |i|
          dot += bufq[i].to_f * buft[i].to_f
          nq += bufq[i].to_f ** 2
          nt += buft[i].to_f ** 2
        end
        nq.should be > 0.0
        cos = dot / (Math.sqrt(nq) * Math.sqrt(nt) + 1e-30)
        # Two quantizations of one weight stay well aligned even at 1.75 bits (measured: IQ1_M
        # 0.884, IQ2_XXS 0.935, IQ4_XS 0.989). A misread field lands near zero, so this
        # threshold separates "lossy" from "wrong" with a wide margin on both sides.
        cos.should be > 0.80
      end
    end
  end

  it "declares the block sizes ggml-common.h specifies" do
    IQ_CASES.each do |t, bytes|
      bs, vals = SHAInet::GGUF::BLOCK_SIZE[t]
      bs.should eq bytes
      vals.should eq 256
    end
  end

  IQ_CASES.each do |type, block_bytes|
    it "computes #{type} the same on the GPU as the scalar reference, on real weights" do
      pending! "CUDA unavailable" unless SHAInet::CUDA.fully_available?
      pending! "CPU kernels unavailable" unless SHAInet::CPUKernels.available?
      pending! "model not present" unless File.exists?(MODEL_IQ)

      gf = SHAInet::GGUF::File.open(MODEL_IQ)
      found = gf.tensors.find { |_, i| i.type == type }
      pending! "no #{type} tensor in the model" unless found
      info = found.not_nil![1]

      k = info.shape[0].to_i32
      n = info.shape[1].to_i32
      rows = n > 48 ? 48 : n # enough to catch a layout error, cheap for a scalar reference
      ptr = gf.tensor_ptr(info).not_nil!
      bytes_per_row = (k // 256) * block_bytes

      x = SHAInet::CudaMatrix.new(1, k)
      # Sign-varied: every one of these formats stores magnitudes and applies signs separately, so an
      # all-positive activation would mask a sign-decoding error entirely.
      k.times { |i| x.raw_data[i] = ((i * 11 % 17) - 8) * 0.03_f32 }
      x.mark_host_modified!
      x.sync_to_device!("iq_x")

      w = SHAInet::GGUFMatrix.new(k, rows, type, ptr, (bytes_per_row.to_u64 * rows))
      dst = SHAInet::CudaMatrix.new(1, rows)
      w.gemv_into(x, dst)
      SHAInet::CUDA.device_synchronize
      dst.mark_device_dirty!
      dst.sync_from_device!("iq_y")

      wbuf = Pointer(Float32).malloc(k)
      worst = 0.0
      rows.times do |r|
        SHAInet::CPUKernels.dequant_row(type.value.to_i32, ptr + r.to_i64 * bytes_per_row, wbuf, k).should be_true
        want = 0.0
        k.times { |i| want += wbuf[i].to_f * x.raw_data[i].to_f }
        got = dst.raw_data[r].to_f
        rel = (got - want).abs / (1.0 + want.abs)
        worst = rel if rel > worst
      end
      # fp32 accumulated in a different order on each side, so compare relatively.
      worst.should be < 1e-3

      # Vacuity: zeros would satisfy any comparison against a reference that also produced zeros.
      nonzero = rows.times.count { |r| dst.raw_data[r].abs > 1e-6 }
      nonzero.should be > rows // 2

      x.free!
      dst.free!
      w.free!
    end
  end
end
