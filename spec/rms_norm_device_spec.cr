require "./spec_helper"

# RMSNorm had no device kernel at all, and its CudaMatrix overload used to read the
# row back, normalise on the host and push it again. That made it impossible to keep
# a block's activation on the device, so these specs pin the kernel's correctness
# before anything is built on top of it.
private def norm_ready?
  SHAInet::CUDA.fully_available? && SHAInet::CUDA.block_device_kernels_available?
end

private def pattern!(m, seed : Float64)
  m.rows.times do |i|
    m.cols.times do |j|
      m[i, j] = Math.sin(seed + i * 0.41 + j * 0.17) * 1.5
    end
  end
  m
end

describe "rms_norm kernel" do
  it "matches an independent CPU reference across multiple rows" do
    pending! "CUDA with the RMSNorm kernel not available" unless norm_ready?

    rows = 5
    cols = 320
    eps = 1e-6
    x = pattern!(SHAInet::CudaMatrix.new(rows, cols), 0.3)
    # Non-uniform gamma: with gamma all ones a dropped multiply would pass unnoticed.
    gamma = SHAInet::CudaMatrix.new(1, cols)
    cols.times { |j| gamma[0, j] = 0.5 + (j % 7) * 0.25 }
    dst = SHAInet::CudaMatrix.new(rows, cols)

    x.mark_host_modified!
    gamma.mark_host_modified!
    x.sync_to_device!("spec_x")
    gamma.sync_to_device!("spec_gamma")

    SHAInet::CUDA.rms_norm_forward(dst.device_ptr.not_nil!, x.device_ptr.not_nil!,
      gamma.device_ptr.not_nil!, rows, cols, eps.to_f32)
    dst.mark_device_dirty!
    dst.sync_from_device!("spec_dst")

    # Independent reference. Several rows on purpose: one row cannot catch a row-stride
    # error, and every row has a different scale so a shared-reduction bug that leaks
    # one row's sum into another would show up.
    rows.times do |i|
      sq = 0.0
      cols.times { |j| v = Math.sin(0.3 + i * 0.41 + j * 0.17) * 1.5; sq += v * v }
      rms = Math.sqrt(sq / cols + eps)
      cols.times do |j|
        v = Math.sin(0.3 + i * 0.41 + j * 0.17) * 1.5
        expected = (v / rms) * (0.5 + (j % 7) * 0.25)
        dst[i, j].should be_close(expected, 1e-4)
      end
    end
  end

  it "handles a column count that is not a multiple of the block size" do
    pending! "CUDA with the RMSNorm kernel not available" unless norm_ready?

    # 256 threads per block with a strided loop: 300 exercises the ragged tail, where
    # an off-by-one in the stride would drop or double-count elements.
    cols = 300
    x = pattern!(SHAInet::CudaMatrix.new(1, cols), 1.1)
    gamma = SHAInet::CudaMatrix.new(1, cols)
    cols.times { |j| gamma[0, j] = 1.0 }
    dst = SHAInet::CudaMatrix.new(1, cols)
    x.mark_host_modified!
    gamma.mark_host_modified!
    x.sync_to_device!("spec_x")
    gamma.sync_to_device!("spec_gamma")

    SHAInet::CUDA.rms_norm_forward(dst.device_ptr.not_nil!, x.device_ptr.not_nil!,
      gamma.device_ptr.not_nil!, 1, cols, 1e-6_f32)
    dst.mark_device_dirty!
    dst.sync_from_device!("spec_dst")

    sq = 0.0
    cols.times { |j| v = Math.sin(1.1 + j * 0.17) * 1.5; sq += v * v }
    rms = Math.sqrt(sq / cols + 1e-6)
    cols.times do |j|
      expected = (Math.sin(1.1 + j * 0.17) * 1.5) / rms
      dst[0, j].should be_close(expected, 1e-4)
    end
  ensure
    x.try(&.free!); gamma.try(&.free!); dst.try(&.free!)
  end
end

describe "add_inplace kernel" do
  it "accumulates elementwise" do
    pending! "CUDA with the residual kernel not available" unless norm_ready?

    n = 513 # deliberately not a multiple of 256, to catch a ragged-tail bug
    a = SHAInet::CudaMatrix.new(1, n)
    b = SHAInet::CudaMatrix.new(1, n)
    n.times { |j| a[0, j] = j.to_f * 0.5; b[0, j] = 100.0 - j.to_f }
    a.mark_host_modified!
    b.mark_host_modified!
    a.sync_to_device!("spec_a")
    b.sync_to_device!("spec_b")

    SHAInet::CUDA.add_inplace(a.device_ptr.not_nil!, b.device_ptr.not_nil!, n)
    a.mark_device_dirty!
    a.sync_from_device!("spec_a_out")

    n.times do |j|
      a[0, j].should be_close(j.to_f * 0.5 + (100.0 - j.to_f), 1e-5)
    end
  ensure
    a.try(&.free!); b.try(&.free!)
  end
end

describe SHAInet::RMSNorm do
  it "the device path matches the host path" do
    pending! "CUDA with the RMSNorm kernel not available" unless norm_ready?

    cols = 256
    norm = SHAInet::RMSNorm.new(cols, 1e-6)
    g = norm.gamma.as(SHAInet::SimpleMatrix)
    cols.times { |j| g[0, j] = 0.75 + (j % 5) * 0.1 }

    host_in = pattern!(SHAInet::SimpleMatrix.new(2, cols), 0.9)
    host_out = norm.forward(host_in)

    norm.to_gpu!
    norm.device_capable?.should be_true

    dev_in = SHAInet::CudaMatrix.new(2, cols)
    2.times { |i| cols.times { |j| dev_in[i, j] = host_in[i, j] } }
    dev_in.mark_host_modified!
    dev_in.sync_to_device!("spec_in")

    dev_out = norm.forward(dev_in)
    dev_out.sync_from_device!("spec_out") if dev_out.device_dirty?

    2.times do |i|
      cols.times { |j| dev_out[i, j].should be_close(host_out[i, j], 1e-4) }
    end
  ensure
    dev_in.try(&.free!); dev_out.try(&.free!)
  end

  it "forward_into writes the caller's destination without allocating a result" do
    pending! "CUDA with the RMSNorm kernel not available" unless norm_ready?

    cols = 128
    norm = SHAInet::RMSNorm.new(cols, 1e-6)
    norm.to_gpu!

    x = pattern!(SHAInet::CudaMatrix.new(1, cols), 0.55)
    x.mark_host_modified!
    x.sync_to_device!("spec_in")
    dst = SHAInet::CudaMatrix.new(1, cols)

    returned = norm.forward_into(x, dst)
    # Must be the SAME object: the device chain reuses one workspace per layer, so a
    # method that quietly allocated its own result would leak one buffer per token.
    returned.should be(dst)
    dst.device_dirty?.should be_true

    dst.sync_from_device!("spec_out")
    sq = 0.0
    cols.times { |j| v = Math.sin(0.55 + j * 0.17) * 1.5; sq += v * v }
    rms = Math.sqrt(sq / cols + 1e-6)
    cols.times do |j|
      dst[0, j].should be_close((Math.sin(0.55 + j * 0.17) * 1.5) / rms, 1e-4)
    end
  ensure
    x.try(&.free!); dst.try(&.free!)
  end
end
