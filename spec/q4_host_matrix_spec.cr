require "./spec_helper"

describe SHAInet::Q4HostMatrix do
  it "produces byte-identical gemv to a device-resident Q4CudaMatrix" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    rng = Random.new(13)
    k = 256
    n = 128

    w = SHAInet::SimpleMatrix.new(k, n)
    k.times { |r| n.times { |c| w[r, c] = (rng.rand * 2.0 - 1.0) } }
    x = SHAInet::SimpleMatrix.new(1, k)
    k.times { |c| x[0, c] = (rng.rand * 2.0 - 1.0) }

    device = SHAInet::Q4CudaMatrix.from_simple(w)
    host = SHAInet::Q4HostMatrix.from_simple(w)

    yd = device.gemv(x.to_cuda); yd.sync_from_device!
    yh = host.gemv(x.to_cuda); yh.sync_from_device!

    # Same packing + same kernel -> exactly equal, not just close.
    n.times do |c|
      yh[0, c].should eq(yd[0, c])
    end
  end

  it "reports zero resident device bytes (weights live in host RAM)" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    w = SHAInet::SimpleMatrix.new(512, 256)
    host = SHAInet::Q4HostMatrix.from_simple(w)
    host.device_bytes.should eq(0_u64)
    host.host_bytes.should be > 0_u64
  end

  it "stores weights in pinned host memory when CUDA is available" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    w = SHAInet::SimpleMatrix.new(256, 128)
    host = SHAInet::Q4HostMatrix.from_simple(w)
    host.pinned?.should be_true
  end

  it "shares one scratch buffer across many experts of the same shape" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    rng = Random.new(21)
    k = 128
    n = 64
    x = SHAInet::SimpleMatrix.new(1, k)
    k.times { |c| x[0, c] = (rng.rand * 2.0 - 1.0) }

    # Two distinct "experts" of identical shape must each yield their own result
    # despite reusing the shared scratch buffer (serialized upload->gemv).
    w1 = SHAInet::SimpleMatrix.new(k, n)
    w2 = SHAInet::SimpleMatrix.new(k, n)
    k.times { |r| n.times { |c| w1[r, c] = rng.rand; w2[r, c] = rng.rand * 2.0 - 1.0 } }

    h1 = SHAInet::Q4HostMatrix.from_simple(w1)
    h2 = SHAInet::Q4HostMatrix.from_simple(w2)
    d1 = SHAInet::Q4CudaMatrix.from_simple(w1)
    d2 = SHAInet::Q4CudaMatrix.from_simple(w2)

    y1 = h1.gemv(x.to_cuda); y1.sync_from_device!
    y2 = h2.gemv(x.to_cuda); y2.sync_from_device!
    r1 = d1.gemv(x.to_cuda); r1.sync_from_device!
    r2 = d2.gemv(x.to_cuda); r2.sync_from_device!

    n.times do |c|
      y1[0, c].should eq(r1[0, c])
      y2[0, c].should eq(r2[0, c])
    end
  end
end
