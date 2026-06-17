require "./simple_matrix"
require "./quantized_cuda_matrix" # QuantizedWeight
require "./q4_cuda_matrix"        # Q4CudaMatrix (packing + GPU scratch)
{% if flag?(:enable_cuda) %}
  require "../cuda"
{% else %}
  require "../cuda_stub"
{% end %}

module SHAInet
  # Host-resident Q4 weight matrix for MoE expert offload.
  #
  # Identical Q4 numerics to Q4CudaMatrix (same packing, same kernel — zero
  # quality loss), but the packed 4-bit weights + fp32 scales live in *host*
  # RAM instead of on the GPU. On each `gemv` the weights are uploaded into a
  # small, shape-keyed GPU scratch buffer shared across all offloaded experts,
  # then the standard Q4 kernel runs.
  #
  # The host buffers are allocated as **pinned (page-locked)** memory when
  # possible, which roughly doubles host->device bandwidth (~6 -> ~12 GB/s on a
  # laptop PCIe4 x8) versus pageable memory and enables async copies. If pinning
  # fails (e.g. the pinned pool is exhausted), it falls back to a pageable array
  # for that weight so the load still succeeds.
  #
  # This lets a large MoE (e.g. Qwen3-Coder-30B-A3B: ~29B params in experts)
  # keep its experts in cheap host RAM while only the few active experts per
  # token ever touch the GPU, so the model fits a 16GB card with no precision
  # change — trading some PCIe bandwidth for VRAM.
  class Q4HostMatrix
    include QuantizedWeight

    getter rows : Int32 # K (in_features)
    getter cols : Int32 # N (out_features)
    getter? pinned : Bool

    @q_ptr : Pointer(UInt8)
    @s_ptr : Pointer(Float32)
    @q_bytes : UInt64
    @s_bytes : UInt64
    # Fallback pageable storage, kept alive only when pinning failed (otherwise
    # the source arrays are freed after the copy into pinned memory).
    @q_arr : Array(UInt8)?
    @s_arr : Array(Float32)?

    # Shared GPU scratch buffers, keyed by [K, N]. Reused across every offloaded
    # expert: experts in a layer share weight shapes, so this holds only a couple
    # of small Q4CudaMatrix buffers regardless of expert count.
    @@scratch = Hash(Tuple(Int32, Int32), Q4CudaMatrix).new
    # The scratch buffer is overwritten by the upload before each GEMV, so the
    # upload+GEMV enqueue must be atomic: a concurrent call (e.g. parallel
    # requests under preview_mt) could otherwise upload a different expert
    # between this call's upload and its kernel launch and corrupt the result.
    @@scratch_mutex = Mutex.new

    def initialize(@rows : Int32, @cols : Int32, q_host : Array(UInt8), s_host : Array(Float32))
      @q_bytes = q_host.size.to_u64
      @s_bytes = s_host.size.to_u64 * 4_u64

      qp = Pointer(UInt8).null
      sp = Pointer(Float32).null
      pinned = false
      if CUDA.fully_available?
        begin
          qpp = Pointer(Void).null
          spp = Pointer(Void).null
          CUDA.malloc_host(pointerof(qpp), @q_bytes)
          CUDA.malloc_host(pointerof(spp), @s_bytes)
          qp = qpp.as(Pointer(UInt8))
          sp = spp.as(Pointer(Float32))
          qp.copy_from(q_host.to_unsafe, q_host.size)
          sp.copy_from(s_host.to_unsafe, s_host.size)
          pinned = true
        rescue
          # Pinned pool exhausted — fall back to pageable for this weight.
          CUDA.free_host(qp.as(Pointer(Void))) unless qp.null?
          qp = Pointer(UInt8).null
          sp = Pointer(Float32).null
          pinned = false
        end
      end

      if pinned
        @q_ptr = qp
        @s_ptr = sp
      else
        @q_arr = q_host
        @s_arr = s_host
        @q_ptr = q_host.to_unsafe
        @s_ptr = s_host.to_unsafe
      end
      @pinned = pinned
    end

    def finalize
      if @pinned
        CUDA.free_host(@q_ptr.as(Pointer(Void))) unless @q_ptr.null?
        CUDA.free_host(@s_ptr.as(Pointer(Void))) unless @s_ptr.null?
      end
    end

    # Quantize a host fp32 weight [K, N] into Q4 and keep it resident in host RAM.
    def self.from_simple(w : SimpleMatrix) : Q4HostMatrix
      q_host, s_host = Q4CudaMatrix.pack(w)
      new(w.rows, w.cols, q_host, s_host)
    end

    # Host memory footprint in bytes (4-bit weights + fp32 scales).
    def host_bytes : UInt64
      @q_bytes + @s_bytes
    end

    # Resident GPU footprint is ~0 — only the shared scratch (counted once,
    # not per-weight) ever lives on the device.
    def device_bytes : UInt64
      0_u64
    end

    private def scratch : Q4CudaMatrix
      (@@scratch[{@rows, @cols}] ||= Q4CudaMatrix.new(@rows, @cols))
    end

    # Copy this expert's packed weights into the shared scratch (host->device).
    private def upload_to(s : Q4CudaMatrix)
      CUDA.memcpy(s.q_ptr.as(Pointer(Void)), @q_ptr.as(Pointer(Void)), @q_bytes, CUDA::MemcpyKind::HostToDevice)
      CUDA.memcpy(s.scale_ptr.as(Pointer(Void)), @s_ptr.as(Pointer(Void)), @s_bytes, CUDA::MemcpyKind::HostToDevice)
    end

    # Upload this expert's weights into the shared scratch and run the Q4 GEMV.
    # Upload + GEMV are enqueued under a mutex so the shared scratch can't be
    # overwritten by a concurrent call between them.
    def gemv(x : CudaMatrix) : CudaMatrix
      @@scratch_mutex.synchronize do
        s = scratch
        upload_to(s)
        s.gemv(x)
      end
    end

    def gemv_into(x : CudaMatrix, result : CudaMatrix) : CudaMatrix
      @@scratch_mutex.synchronize do
        s = scratch
        upload_to(s)
        s.gemv_into(x, result)
      end
    end
  end
end
