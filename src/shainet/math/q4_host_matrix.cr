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
  # Two throughput optimizations layer on top, both transparent and lossless:
  #   * Host buffers are **pinned** (page-locked) when possible (~2x H2D vs
  #     pageable), falling back to a pageable array if the pinned pool is full.
  #   * A global **hot-expert cache** keeps frequently-used weights resident on
  #     the GPU (as Q4CudaMatrix) within a VRAM budget, so the hottest experts
  #     skip the host->device upload entirely. MoE routing is skewed, so a
  #     budget well under the full expert set still yields a high hit rate.
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
    @q_arr : Array(UInt8)?
    @s_arr : Array(Float32)?

    # Shape-keyed GPU scratch for cold (uncached) experts. Reused across experts.
    @@scratch = Hash(Tuple(Int32, Int32), Q4CudaMatrix).new
    # One mutex guards all shared GPU state (scratch + cache) so concurrent
    # gemv calls can't corrupt the shared scratch or the cache bookkeeping.
    @@gpu_mutex = Mutex.new

    # --- Hot-expert cache (LRU, frequency-gated) -------------------------------
    # Insertion-ordered map = LRU order (front = least recently used). A weight is
    # promoted to a resident Q4CudaMatrix only after it has been used
    # PROMOTE_THRESHOLD times, so one-off cold experts never evict hot ones.
    @@resident = Hash(Q4HostMatrix, Q4CudaMatrix).new
    @@freq = Hash(Q4HostMatrix, Int32).new(0)
    @@used_bytes = 0_u64
    @@budget_bytes : UInt64? = nil
    @@hits = 0_u64
    @@misses = 0_u64
    PROMOTE_THRESHOLD = 2

    def initialize(@rows : Int32, @cols : Int32, q_host : Array(UInt8), s_host : Array(Float32))
      @q_bytes = q_host.size.to_u64
      @s_bytes = s_host.size.to_u64 * 4_u64

      qp = Pointer(UInt8).null
      sp = Pointer(Float32).null
      pinned = false
      if CUDA.fully_available?
        qpp = Pointer(Void).null
        spp = Pointer(Void).null
        begin
          CUDA.malloc_host(pointerof(qpp), @q_bytes)
          CUDA.malloc_host(pointerof(spp), @s_bytes)
          qp = qpp.as(Pointer(UInt8))
          sp = spp.as(Pointer(Float32))
          qp.copy_from(q_host.to_unsafe, q_host.size)
          sp.copy_from(s_host.to_unsafe, s_host.size)
          pinned = true
        rescue
          # Free whatever was allocated before the failure (either pointer may be
          # set independently of qp/sp), then fall back to pageable for this weight.
          CUDA.free_host(qpp) unless qpp.null?
          CUDA.free_host(spp) unless spp.null?
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

    def self.from_simple(w : SimpleMatrix) : Q4HostMatrix
      q_host, s_host = Q4CudaMatrix.pack(w)
      new(w.rows, w.cols, q_host, s_host)
    end

    # Host memory footprint in bytes (4-bit weights + fp32 scales).
    def host_bytes : UInt64
      @q_bytes + @s_bytes
    end

    # Resident GPU footprint per-weight is ~0 unless this weight is currently in
    # the shared hot cache; the cache is budgeted globally (see cache_stats).
    def device_bytes : UInt64
      0_u64
    end

    # GPU memory budget for the hot-expert cache. Defaults to 70% of free VRAM
    # at first use (leaving room for activations / KV cache / scratch), or set
    # SHAINET_EXPERT_CACHE_MB (0 disables caching).
    def self.budget_bytes : UInt64
      @@budget_bytes ||= begin
        if mb = ENV["SHAINET_EXPERT_CACHE_MB"]?
          mb.to_u64 * 1024_u64 * 1024_u64
        elsif info = CUDA.memory_info
          (info[:free].to_f * 0.70).to_u64
        else
          0_u64
        end
      end
    end

    def self.cache_stats : NamedTuple(resident: Int32, used_mb: Float64, budget_mb: Float64, hits: UInt64, misses: UInt64, hit_rate: Float64)
      @@gpu_mutex.synchronize do
        total = @@hits + @@misses
        {
          resident:  @@resident.size,
          used_mb:   (@@used_bytes / 1024.0 / 1024.0),
          budget_mb: (budget_bytes / 1024.0 / 1024.0),
          hits:      @@hits,
          misses:    @@misses,
          hit_rate:  total.zero? ? 0.0 : (@@hits.to_f / total),
        }
      end
    end

    private def upload_to(s : Q4CudaMatrix)
      CUDA.memcpy(s.q_ptr.as(Pointer(Void)), @q_ptr.as(Pointer(Void)), @q_bytes, CUDA::MemcpyKind::HostToDevice)
      CUDA.memcpy(s.scale_ptr.as(Pointer(Void)), @s_ptr.as(Pointer(Void)), @s_bytes, CUDA::MemcpyKind::HostToDevice)
    end

    private def scratch : Q4CudaMatrix
      (@@scratch[{@rows, @cols}] ||= Q4CudaMatrix.new(@rows, @cols))
    end

    # Resolve the device matrix to run this GEMV on (must hold @@gpu_mutex):
    #   * cache hit  -> the resident Q4CudaMatrix (no upload), marked MRU
    #   * cache miss -> upload into the shared scratch; promote to a resident
    #     copy when the weight is hot enough and the budget allows.
    # Caching is best-effort and transparent: any failure degrades to the scratch
    # path rather than crashing inference.
    private def device_matrix : Q4CudaMatrix
      # Caching disabled (budget 0) -> always use the shared scratch.
      if Q4HostMatrix.budget_bytes == 0
        s = scratch
        upload_to(s)
        return s
      end

      if r = @@resident[self]?
        # LRU touch: reinsert to move to the most-recently-used end.
        @@resident.delete(self)
        @@resident[self] = r
        @@hits += 1
        return r
      end

      @@misses += 1
      @@freq[self] += 1
      if @@freq[self] >= PROMOTE_THRESHOLD && admit?
        if promoted = promote
          return promoted
        end
      end

      s = scratch
      upload_to(s)
      s
    end

    # Create a resident GPU copy of this weight and record it. Returns nil (and
    # leaves the cache untouched) if allocation/upload fails under VRAM pressure,
    # so the caller falls back to the scratch path.
    private def promote : Q4CudaMatrix?
      r = Q4CudaMatrix.new(@rows, @cols)
      upload_to(r)
      @@resident[self] = r
      @@used_bytes += r.device_bytes
      @@freq.delete(self) # promoted: drop the freq entry (and its strong ref)
      r
    rescue
      r.free! if r
      nil
    end

    # Make room for a resident copy of this weight within budget by evicting the
    # least-recently-used residents. Returns false if it can never fit.
    private def admit? : Bool
      need = Q4CudaMatrix.device_bytes_for(@rows, @cols)
      budget = Q4HostMatrix.budget_bytes
      return false if need > budget
      while @@used_bytes + need > budget && !@@resident.empty?
        victim_k, victim_v = @@resident.first
        @@resident.delete(victim_k)
        @@freq.delete(victim_k) # don't retain freq (and a strong ref) for evicted weights
        @@used_bytes -= victim_v.device_bytes
        victim_v.free! # reclaim VRAM immediately, don't wait for GC
      end
      @@used_bytes + need <= budget
    end

    def gemv(x : CudaMatrix) : CudaMatrix
      @@gpu_mutex.synchronize { device_matrix.gemv(x) }
    end

    def gemv_into(x : CudaMatrix, result : CudaMatrix) : CudaMatrix
      @@gpu_mutex.synchronize { device_matrix.gemv_into(x, result) }
    end
  end
end
