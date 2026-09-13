require "./spec_helper"

# The Q4 GEMM has two kernels behind one entry point: a vectorized one (one warp
# per output column, uint4 weight loads, warp-shuffle reduction) used when
# K % 32 == 0, and the original scalar kernel as a fallback for shapes whose
# packed rows are not 16-byte aligned.
#
# These specs check the kernels against an INDEPENDENT CPU implementation of the
# documented packing, not against fp32. Comparing to fp32 only bounds
# quantization error and would not notice a mis-indexed nibble, a wrong scale
# block, or a lane that silently drops its slice.
private def q4_reference(w : SHAInet::SimpleMatrix, x : SHAInet::SimpleMatrix) : SHAInet::SimpleMatrix
  k = w.rows
  n = w.cols
  m = x.rows
  q_host, d_host, sub_host = SHAInet::Q4CudaMatrix.pack(w)
  nblocks = (k + 31) // 32
  nsupers = (nblocks + SHAInet::Q4CudaMatrix::SUPER - 1) // SHAInet::Q4CudaMatrix::SUPER
  kbytes = (k + 1) // 2

  out = SHAInet::SimpleMatrix.new(m, n)
  m.times do |row|
    n.times do |col|
      acc = 0.0
      k.times do |kk|
        byte = q_host[col * kbytes + (kk >> 1)]
        nib = (kk & 1) == 1 ? (byte >> 4) : (byte & 0x0F)
        b = kk >> 5
        # Effective block scale: fp32 super-block scale times the block's byte
        # sub-scale over 255.
        eff = d_host[col * nsupers + (b // SHAInet::Q4CudaMatrix::SUPER)].to_f64 *
              sub_host[col * nblocks + b].to_f64 / 255.0
        acc += (nib.to_i - 8) * x[row, kk] * eff
      end
      out[row, col] = acc
    end
  end
  out
end

private def random_matrix(rows : Int32, cols : Int32, seed : Int32) : SHAInet::SimpleMatrix
  m = SHAInet::SimpleMatrix.new(rows, cols)
  rng = Random.new(seed)
  rows.times { |r| cols.times { |c| m[r, c] = (rng.rand * 2.0 - 1.0) } }
  m
end

# Largest relative deviation, scaled by the magnitude of the reference row so a
# near-zero output does not blow the ratio up.
private def max_rel_error(got : SHAInet::SimpleMatrix, want : SHAInet::SimpleMatrix) : Float64
  scale = 0.0
  want.rows.times { |r| want.cols.times { |c| a = want[r, c].abs; scale = a if a > scale } }
  scale = 1.0 if scale == 0.0
  worst = 0.0
  want.rows.times do |r|
    want.cols.times do |c|
      d = (got[r, c] - want[r, c]).abs / scale
      worst = d if d > worst
    end
  end
  worst
end

private def check_shape(k : Int32, n : Int32, m : Int32, seed : Int32)
  w = random_matrix(k, n, seed)
  x = random_matrix(m, k, seed + 1)
  want = q4_reference(w, x)

  qw = SHAInet::Q4CudaMatrix.from_simple(w)
  got = qw.gemv(x.to_cuda)
  got.sync_from_device!

  got.rows.should eq m
  got.cols.should eq n
  # Same arithmetic, different accumulation order, so this is float reassociation
  # only. Anything structural (wrong nibble, wrong scale, dropped lane) lands far
  # outside this bound.
  max_rel_error(got.to_simple, want).should be < 1.0e-4
end

describe "Q4 GEMM kernel" do
  it "matches the packed reference on the vectorized path (decode, M=1)" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    check_shape(2048, 64, 1, 3) # K % 32 == 0
  end

  it "matches the packed reference on the vectorized path (prefill, M>1)" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    check_shape(2048, 64, 4, 5)
  end

  it "matches the packed reference on the scalar fallback path" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    check_shape(255, 17, 1, 7) # K % 32 != 0 -> unaligned packed rows -> fallback
  end

  it "handles a K spanning several shared-memory tiles with a partial last tile" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    # Q4_TILE is 1024 activations, so 1056 gives a full tile plus a 32-wide tail
    # in which only lane 0 has work. That tail is where an off-by-one in the
    # per-lane guard would hide.
    check_shape(1056, 24, 2, 11)
  end

  it "handles an output width that is not a multiple of the columns per block" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    # 8 warps per block, so N=13 leaves 3 warps of the last block with no column.
    check_shape(512, 13, 1, 13)
  end

  it "handles a single output column, where only one warp has work" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    check_shape(32, 1, 1, 17) # exactly one scale block, one column
  end

  it "is faster than the scalar fallback at equivalent work", tags: "perf" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    # A wide N so the kernel dominates. At small N the fixed launch cost swamps
    # the difference (measured 1.06x at N=512 versus 1.95x at N=151936), which
    # would make this spec a coin flip rather than a guard.
    n = 16_384
    # K=2048 takes the vectorized path, K=2047 the scalar one. Within 0.05% the
    # same arithmetic, so this is a direct A/B of the two kernels.
    vec_w = SHAInet::Q4CudaMatrix.from_simple(random_matrix(2048, n, 21))
    sca_w = SHAInet::Q4CudaMatrix.from_simple(random_matrix(2047, n, 21))
    vec_x = random_matrix(1, 2048, 22).to_cuda
    sca_x = random_matrix(1, 2047, 22).to_cuda
    dst = SHAInet::CudaMatrix.new(1, n)

    # gemv_into with a preallocated destination, synced once per batch: gemv
    # allocates a fresh result matrix and a per-call sync_from_device! adds a D2H
    # round trip, and together those cost more than the kernel at these sizes.
    best = ->(w : SHAInet::Q4CudaMatrix, xx : SHAInet::CudaMatrix) do
      iters = 50
      w.gemv_into(xx, dst)
      dst.sync_from_device! # warm up
      lo = Time::Span::MAX
      3.times do
        t0 = Time.monotonic
        iters.times { w.gemv_into(xx, dst) }
        dst.sync_from_device!
        dt = (Time.monotonic - t0) / iters
        lo = dt if dt < lo
      end
      lo
    end

    vec = best.call(vec_w, vec_x)
    sca = best.call(sca_w, sca_x)
    # Measured ~0.5-0.65 at this width. If this ever climbs back toward parity
    # the vectorized path has been disabled or undone.
    (vec.total_milliseconds / sca.total_milliseconds).should be < 0.8
  end
end
