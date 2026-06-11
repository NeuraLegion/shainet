require "./spec_helper"

describe SHAInet::QuantizedCudaMatrix do
  it "matches fp32 matmul within Q8 quantization error" do
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

    qw = SHAInet::QuantizedCudaMatrix.from_simple(w)
    y = qw.gemv(x.to_cuda)
    y.sync_from_device!

    # Cosine similarity between quantized and fp32 results should be ~1.
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
    cos.should be > 0.999
  end

  it "exposes a ~4x smaller device footprint than fp32" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    k = 2048
    n = 512
    w = SHAInet::SimpleMatrix.new(k, n)
    qw = SHAInet::QuantizedCudaMatrix.from_simple(w)

    fp32_bytes = (k.to_u64 * n.to_u64 * 4_u64)
    # int8 weights + small fp32 scales should be well under half of fp32.
    qw.device_bytes.should be < (fp32_bytes // 2)
  end
end
