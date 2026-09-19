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

# type => {block bytes, reference dequant}
IQ_CASES = {
  SHAInet::GGUF::GGMLType::IQ4_XS  => 136,
  SHAInet::GGUF::GGMLType::IQ3_S   => 110,
  SHAInet::GGUF::GGMLType::IQ3_XXS => 98,
}

describe "i-quant kernels" do
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
