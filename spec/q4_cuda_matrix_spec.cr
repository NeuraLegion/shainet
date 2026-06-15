require "./spec_helper"

describe SHAInet::Q4CudaMatrix do
  it "matches fp32 matmul within Q4 quantization error" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    rng = Random.new(7)
    k = 256
    n = 128
    m = 1

    w = SHAInet::SimpleMatrix.new(k, n)
    k.times { |r| n.times { |c| w[r, c] = (rng.rand * 2.0 - 1.0) } }
    x = SHAInet::SimpleMatrix.new(m, k)
    m.times { |r| k.times { |c| x[r, c] = (rng.rand * 2.0 - 1.0) } }

    ref = x * w # fp32 reference [m, n]

    qw = SHAInet::Q4CudaMatrix.from_simple(w)
    y = qw.gemv(x.to_cuda)
    y.sync_from_device!

    # Cosine similarity between quantized and fp32 results. 4-bit is coarser than
    # Q8 (16 levels/block) so we allow a looser bound, but it must still track.
    dot = 0.0
    na = 0.0
    nb = 0.0
    m.times do |r|
      n.times do |c|
        a = ref[r, c]
        b = y[r, c]
        dot += a * b
        na += a * a
        nb += b * b
      end
    end
    cos = dot / (Math.sqrt(na) * Math.sqrt(nb))
    cos.should be > 0.99
  end

  it "handles odd K (nibble padding) correctly" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    rng = Random.new(11)
    k = 65 # odd → last byte has a single used nibble
    n = 16
    w = SHAInet::SimpleMatrix.new(k, n)
    k.times { |r| n.times { |c| w[r, c] = (rng.rand * 2.0 - 1.0) } }
    x = SHAInet::SimpleMatrix.new(1, k)
    k.times { |c| x[0, c] = (rng.rand * 2.0 - 1.0) }

    ref = x * w
    qw = SHAInet::Q4CudaMatrix.from_simple(w)
    y = qw.gemv(x.to_cuda)
    y.sync_from_device!

    dot = 0.0; na = 0.0; nb = 0.0
    n.times do |c|
      a = ref[0, c]; b = y[0, c]
      dot += a * b; na += a * a; nb += b * b
    end
    cos = dot / (Math.sqrt(na) * Math.sqrt(nb))
    cos.should be > 0.99
  end

  it "exposes a ~8x smaller device footprint than fp32" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    k = 2048
    n = 512
    w = SHAInet::SimpleMatrix.new(k, n)
    qw = SHAInet::Q4CudaMatrix.from_simple(w)

    fp32_bytes = (k.to_u64 * n.to_u64 * 4_u64)
    # 4-bit weights + small fp32 scales should be well under a quarter of fp32.
    qw.device_bytes.should be < (fp32_bytes // 4)
  end
end
