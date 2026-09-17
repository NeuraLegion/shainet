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
    @sub_ptr : Pointer(UInt8)
    @q_bytes : UInt64
    @s_bytes : UInt64
    @sub_bytes : UInt64
    @q_arr : Array(UInt8)?
    @s_arr : Array(Float32)?
    @sub_arr : Array(UInt8)?

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

    def initialize(@rows : Int32, @cols : Int32, q_host : Array(UInt8), s_host : Array(Float32), sub_host : Array(UInt8))
      @q_bytes = q_host.size.to_u64
      @s_bytes = s_host.size.to_u64 * 4_u64
      @sub_bytes = sub_host.size.to_u64

      qp = Pointer(UInt8).null
      sp = Pointer(Float32).null
      bp = Pointer(UInt8).null
      pinned = false
      if CUDA.fully_available?
        qpp = Pointer(Void).null
        spp = Pointer(Void).null
        bpp = Pointer(Void).null
        begin
          CUDA.malloc_host(pointerof(qpp), @q_bytes)
          CUDA.malloc_host(pointerof(spp), @s_bytes)
          CUDA.malloc_host(pointerof(bpp), @sub_bytes)
          qp = qpp.as(Pointer(UInt8))
          sp = spp.as(Pointer(Float32))
          bp = bpp.as(Pointer(UInt8))
          qp.copy_from(q_host.to_unsafe, q_host.size)
          sp.copy_from(s_host.to_unsafe, s_host.size)
          bp.copy_from(sub_host.to_unsafe, sub_host.size)
          pinned = true
        rescue
          # Free whatever was allocated before the failure (any pointer may be set
          # independently of the others), then fall back to pageable for this weight.
          CUDA.free_host(qpp) unless qpp.null?
          CUDA.free_host(spp) unless spp.null?
          CUDA.free_host(bpp) unless bpp.null?
          qp = Pointer(UInt8).null
          sp = Pointer(Float32).null
          bp = Pointer(UInt8).null
          pinned = false
        end
      end

      if pinned
        @q_ptr = qp
        @s_ptr = sp
        @sub_ptr = bp
      else
        @q_arr = q_host
        @s_arr = s_host
        @sub_arr = sub_host
        @q_ptr = q_host.to_unsafe
        @s_ptr = s_host.to_unsafe
        @sub_ptr = sub_host.to_unsafe
      end
      @pinned = pinned
    end

    def finalize
      if @pinned
        CUDA.free_host(@q_ptr.as(Pointer(Void))) unless @q_ptr.null?
        CUDA.free_host(@s_ptr.as(Pointer(Void))) unless @s_ptr.null?
        CUDA.free_host(@sub_ptr.as(Pointer(Void))) unless @sub_ptr.null?
      end
    end

    def self.from_simple(w : SimpleMatrix) : Q4HostMatrix
      q_host, s_host, sub_host = Q4CudaMatrix.pack(w)
      new(w.rows, w.cols, q_host, s_host, sub_host)
    end

    # Load a pre-quantized weight from the three raw cache files, host-resident (streamed to the
    # device on demand). The dense-offload counterpart to Q4CudaMatrix.from_files.
    def self.from_files(rows : Int32, cols : Int32, q_path : String, d_path : String, sub_path : String) : Q4HostMatrix
      q_bytes = File.read(q_path).to_slice
      d_bytes = File.read(d_path).to_slice
      sub_bytes = File.read(sub_path).to_slice
      q_host = Array(UInt8).new(q_bytes.size) { |i| q_bytes[i] }
      s_host = Array(Float32).new(d_bytes.size // 4) { |i| IO::ByteFormat::LittleEndian.decode(Float32, d_bytes[i * 4, 4]) }
      sub_host = Array(UInt8).new(sub_bytes.size) { |i| sub_bytes[i] }
      new(rows, cols, q_host, s_host, sub_host)
    end

    # Host memory footprint in bytes (4-bit weights + super-block scales + sub-scales).
    def host_bytes : UInt64
      @q_bytes + @s_bytes + @sub_bytes
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

    # Set the hot-cache budget explicitly, overriding the env var and the default.
    #
    # Needed because the default is BOTH process-global and memoized on first use
    # (70% of whatever VRAM happened to be free then), so SHAINET_EXPERT_CACHE_MB
    # cannot steer it after the first weight has been touched. Callers that must
    # bound VRAM deterministically, and specs that must not let one example's
    # cache starve later ones, set it here instead.
    def self.budget_bytes=(bytes : UInt64) : UInt64
      @@gpu_mutex.synchronize { @@budget_bytes = bytes }
      bytes
    end

    # Temporarily uncap the budget for a prefill pass. During prefill every weight is touched
    # exactly once per layer in order, so the LRU fills optimally. The device_has_room? check
    # and the promote rescue are the real guards against over-allocation; the budget is just a
    # soft ceiling that causes unnecessary eviction/streaming during prefill.
    #
    # Call prefill_boost! before the prefill pass, prefill_restore! after. Also lowers the
    # reserve (the floor of free VRAM below which promotion is refused) to a prefill-safe
    # minimum, since prefill workspaces are small and transient.
    @@saved_budget : UInt64? = nil
    @@saved_reserve : UInt64? = nil
    PREFILL_RESERVE_MB = 1024

    def self.prefill_boost!
      @@gpu_mutex.synchronize do
        @@saved_budget = @@budget_bytes
        @@saved_reserve = @@reserve_bytes
        @@budget_bytes = UInt64::MAX
        @@reserve_bytes = PREFILL_RESERVE_MB.to_u64 * 1024_u64 * 1024_u64
      end
    end

    def self.prefill_restore!
      @@gpu_mutex.synchronize do
        if b = @@saved_budget
          @@budget_bytes = b
          @@saved_budget = nil
        end
        if r = @@saved_reserve
          @@reserve_bytes = r
          @@saved_reserve = nil
          # Evict the least-recently-used weights until the device has enough free
          # VRAM for the restored reserve (decode workspaces + KV cache). The budget
          # is NOT the eviction target: we WANT the prefill-promoted weights to stay
          # resident as long as possible, and only shed enough to leave decode headroom.
          if info = CUDA.memory_info
            while info[:free] < r && !@@resident.empty?
              victim_k, victim_v = @@resident.first
              @@resident.delete(victim_k)
              freed = victim_v.device_bytes
              @@used_bytes = @@used_bytes > freed ? @@used_bytes - freed : 0_u64
              victim_v.free!
              info = CUDA.memory_info || info
            end
          end
        end
      end
    end

    # Evict every resident copy and drop the shared scratch, returning that VRAM
    # to the device. The cache is otherwise held for the life of the process.
    def self.release_cache!
      @@gpu_mutex.synchronize do
        @@resident.each_value(&.free!)
        @@resident.clear
        @@scratch.each_value(&.free!)
        @@scratch.clear
        @@freq.clear
        @@used_bytes = 0_u64
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

    # The PCIe cost of a cache miss.
    #
    # NOTE this phase NESTS inside ffn.dev_gate_up and ffn.dev_down, which wrap the expert
    # GEMV that resolves the weight. It is therefore the one phase that breaks the module's
    # partition rule, and the report's total over-counts by exactly this much: measured,
    # phases summed to 98.28 ms/step against 68.33 ms of wall time, and 98.28 - 32.57 is
    # 65.71, which reconciles. Read it as a SUBSET of the expert GEMV phases, never as a
    # sibling of them.
    #
    # It is worth that inconvenience because it splits transfer from arithmetic inside the
    # expert path, and that split is decode's whole question: 8 experts across 48 layers is
    # 384 weight resolutions per token, and measured 283 of them miss, making decode
    # transfer-bound rather than compute-bound.
    private def upload_to(s : Q4CudaMatrix)
      Profile.measure("expert.h2d(nested)") do
        CUDA.memcpy(s.q_ptr.as(Pointer(Void)), @q_ptr.as(Pointer(Void)), @q_bytes, CUDA::MemcpyKind::HostToDevice)
        CUDA.memcpy(s.d_ptr.as(Pointer(Void)), @s_ptr.as(Pointer(Void)), @s_bytes, CUDA::MemcpyKind::HostToDevice)
        CUDA.memcpy(s.sub_ptr.as(Pointer(Void)), @sub_ptr.as(Pointer(Void)), @sub_bytes, CUDA::MemcpyKind::HostToDevice)
      end
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
      # Promote on FIRST touch when this weight fits without evicting anything, and only
      # then fall back to the frequency gate.
      #
      # PROMOTE_THRESHOLD exists so a cold one-off does not displace a hot weight. That is
      # a statement about CONTENTION, and it was being applied unconditionally -- so with
      # VRAM to spare a rarely-touched weight stayed on the streaming path forever, paying
      # PCIe on every single touch while the budget sat idle. Measured on the 30B at a
      # 12000 MB budget: the whole touched working set (7932 MB) was resident with 4068 MB
      # unused, yet 246 of 1152 resolutions per step still missed, costing 33.88 ms of a
      # 64.82 ms step. Nothing needed evicting; the gate was simply refusing free wins.
      #
      # With headroom there is no victim, so the threshold protects nothing and only costs.
      if fits_without_eviction? || (@@freq[self] >= PROMOTE_THRESHOLD && admit?)
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

    # VRAM that first-touch promotion must leave on the device, over and above the budget.
    #
    # Needed because this change made the budget HONEST. Before it, the frequency gate meant
    # the cache only ever reached the touched working set -- measured 7932 MB against a
    # 12000 MB budget -- so an over-large budget was safe by accident. Filling it on first
    # touch removes that accident: a 12000 MB budget on a 16 GB card duly consumed 12000 MB
    # and the next KV/workspace allocation failed with cudaMalloc result 2.
    #
    # The budget is the user's declared ceiling, but nothing else on the device gets a vote
    # in it, and the KV cache is allocated AFTER the weights, growing with context. So this
    # is a floor on free VRAM, checked against the device rather than against the budget.
    # SHAINET_EXPERT_CACHE_RESERVE_MB overrides it.
    RESERVE_MB_DEFAULT = 768

    @@reserve_bytes : UInt64? = nil

    def self.reserve_bytes : UInt64
      v = @@reserve_bytes
      return v if v
      mb = (ENV["SHAINET_EXPERT_CACHE_RESERVE_MB"]? || RESERVE_MB_DEFAULT.to_s).to_i
      mb = 0 if mb < 0
      v = mb.to_u64 * 1024_u64 * 1024_u64
      @@reserve_bytes = v
      v
    end

    def self.reserve_bytes=(bytes : UInt64?)
      @@reserve_bytes = bytes
    end

    # Would a resident copy fit in the remaining budget WITHOUT evicting anything, and
    # without eating into the device reserve?
    #
    # Deliberately side-effect free, unlike `admit?`, which evicts to make room. That split
    # is what lets the caller distinguish "free win" from "contended", and treat only the
    # contended case as needing a frequency gate.
    private def fits_without_eviction? : Bool
      need = Q4CudaMatrix.device_bytes_for(@rows, @cols)
      return false if @@used_bytes + need > Q4HostMatrix.budget_bytes
      device_has_room?(need)
    end

    # Is there room on the DEVICE for `need` more bytes, keeping the reserve intact?
    #
    # Asks the device rather than the budget, because everything else competing for VRAM --
    # the KV cache, prefill workspaces, another process -- is invisible to the budget
    # arithmetic. Applied on BOTH promotion paths: gating only first touch would leave the
    # hole that a weight touched twice still promotes into the reserve.
    private def device_has_room?(need : UInt64) : Bool
      reserve = Q4HostMatrix.reserve_bytes
      return true if reserve == 0
      if info = CUDA.memory_info
        return info[:free] >= need + reserve
      end
      true
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
      # Checked AFTER the evictions above, which call free! immediately, so the device query
      # sees the VRAM they returned rather than a stale reading.
      @@used_bytes + need <= budget && device_has_room?(need)
    end

    def gemv(x : CudaMatrix) : CudaMatrix
      @@gpu_mutex.synchronize { device_matrix.gemv(x) }
    end

    def gemv_into(x : CudaMatrix, result : CudaMatrix) : CudaMatrix
      @@gpu_mutex.synchronize { device_matrix.gemv_into(x, result) }
    end
  end
end
