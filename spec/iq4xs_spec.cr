require "./spec_helper"

# IQ4_XS support, verified against the scalar reference ported from ggml-quants.c.
#
# Two things are checked, and the second matters more than the first: that the CUDA GEMV agrees with
# a dequantize-then-dot computed from the SAME bytes, and that it does so on blocks taken from a real
# GSQ-RCO model file rather than bytes I made up. Synthetic blocks can satisfy a kernel that
# misreads the 6-bit split scale (scales_l nibble plus scales_h 2-bit pair) because a hand-built
# block tends to have small, similar scales; real tensors do not.
MODEL_IQ4XS = "/home/unshadow/models/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"

describe "IQ4_XS" do
  it "has the block size ggml-common.h specifies" do
    bs, vals = SHAInet::GGUF::BLOCK_SIZE[SHAInet::GGUF::GGMLType::IQ4_XS]
    # d(2) + scales_h(2) + scales_l[QK_K/64 = 4] + qs[QK_K/2 = 128]
    bs.should eq 136
    vals.should eq 256
  end

  it "matches the scalar reference in the CUDA GEMV, on real model weights" do
    pending! "CUDA unavailable" unless SHAInet::CUDA.fully_available?
    pending! "CPU kernels unavailable" unless SHAInet::CPUKernels.available?
    pending! "model not present: #{MODEL_IQ4XS}" unless File.exists?(MODEL_IQ4XS)

    gf = SHAInet::GGUF::File.open(MODEL_IQ4XS)
    _name, info = gf.tensors.find { |_, i| i.type.iq4_xs? } || raise "no IQ4_XS tensor in the model"
    k = info.shape[0].to_i32
    n = info.shape[1].to_i32
    # A few hundred output rows is enough to catch a layout error and keeps the reference cheap.
    rows = n > 64 ? 64 : n
    ptr = gf.tensor_ptr(info).not_nil!

    x = SHAInet::CudaMatrix.new(1, k)
    # Sign-varied activations: kvalues_iq4nl is asymmetric, so an all-positive input can hide a
    # mistake in which nibble maps to which value.
    k.times { |i| x.raw_data[i] = ((i * 7 % 13) - 6) * 0.05_f32 }
    x.mark_host_modified!
    x.sync_to_device!("iq4xs_x")

    w = SHAInet::GGUFMatrix.new(k, rows, info.type, ptr, (info.byte_size // n * rows).to_u64)
    dst = SHAInet::CudaMatrix.new(1, rows)
    w.gemv_into(x, dst)
    SHAInet::CUDA.device_synchronize
    dst.mark_device_dirty!
    dst.sync_from_device!("iq4xs_y")

    # Reference: dequantize each row with the ported scalar routine and dot it by hand.
    bytes_per_row = (k // 256) * 136
    wbuf = Pointer(Float32).malloc(k)
    rows.times do |r|
      ok = SHAInet::CPUKernels.dequant_iq4xs_row(ptr + r.to_i64 * bytes_per_row, wbuf, k)
      ok.should be_true
      want = 0.0
      k.times { |i| want += wbuf[i].to_f * x.raw_data[i].to_f }
      got = dst.raw_data[r].to_f
      # fp32 accumulation in a different order; the tolerance scales with the magnitudes involved.
      tol = 1e-3 * (1.0 + want.abs)
      (got - want).abs.should be < tol
    end

    # Vacuity: a kernel that wrote zeros would satisfy a reference that also computed zeros.
    nonzero = rows.times.count { |r| dst.raw_data[r].abs > 1e-6 }
    nonzero.should be > rows // 2

    x.free!
    dst.free!
    w.free!
  end
end
