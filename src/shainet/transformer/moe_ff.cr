require "./swiglu_ff"

module SHAInet
  # Mixture-of-Experts feed-forward, Qwen3-MoE style.
  #
  # A small router projects each token to `num_experts` logits; the top-`top_k`
  # experts (by softmax weight) are evaluated and their outputs combined as a
  # weighted sum. Each expert is a SwiGLU FFN (see SwiGLUFF). Only `top_k` of
  # `num_experts` experts run per token, so a 128-expert / top-8 layer does the
  # work of ~8 dense FFNs while holding all 128 in memory.
  #
  # The router is kept in full precision (it is tiny and routing is sensitive to
  # quantization error); experts are quantizable to Q8/Q4 like any other weight.
  class MoEFF
    getter experts : Array(SwiGLUFF)
    property router : SimpleMatrix | CudaMatrix # [d_model, num_experts]
    getter num_experts : Int32
    getter top_k : Int32
    getter? norm_topk_prob : Bool
    getter d_model : Int32
    # When true, experts are quantized to host-resident Q4 (Q4HostMatrix) and
    # streamed to the GPU on demand, instead of living on the GPU. Lets large
    # MoE models fit small cards at the cost of PCIe bandwidth. The router stays
    # on the GPU regardless.
    property? offload_experts : Bool = false

    def initialize(@d_model : Int32, ff_hidden : Int32, @num_experts : Int32,
                   @top_k : Int32, @norm_topk_prob : Bool = true, @offload_experts : Bool = false)
      raise ArgumentError.new("num_experts must be positive") unless @num_experts > 0
      raise ArgumentError.new("top_k must be in 1..num_experts") unless 1 <= @top_k <= @num_experts
      @router = SimpleMatrix.new(@d_model, @num_experts)
      # Experts start as empty placeholders; the loader assigns real weights and
      # quantizes them per layer, so we never hold all experts' fp32 at once.
      @experts = Array(SwiGLUFF).new(@num_experts) { SwiGLUFF.new(@d_model, ff_hidden, allocate: false) }
    end

    def to_gpu!(quantize : Bool = false, bits : Int32 = 8)
      return unless CUDA.fully_available?
      # Router stays full precision (CudaMatrix); only experts are quantized
      # (and, when offload_experts is set, kept host-resident).
      @router = @router.as(SimpleMatrix).to_cuda if @router.is_a?(SimpleMatrix)
      @experts.each(&.to_gpu!(quantize, bits, @offload_experts))
    end

    # Router logits [tokens, num_experts] for the given activations.
    private def router_logits(x : SimpleMatrix) : SimpleMatrix
      r = @router
      if r.is_a?(CudaMatrix)
        xg = x.to_cuda
        prod = xg * r
        res = prod.to_simple
        xg.free!
        prod.free!
        res
      else
        x * r.as(SimpleMatrix)
      end
    end

    # Top-k (expert_index, weight) for row `t`: softmax over all experts, take
    # the top-k by probability, then (when norm_topk_prob) renormalize the
    # selected weights so they sum to 1 — matching the Qwen3-MoE router.
    private def top_k_gating(logits : SimpleMatrix, t : Int32) : Array(Tuple(Int32, Float64))
      ne = @num_experts
      maxl = -Float64::INFINITY
      ne.times { |e| v = logits[t, e].to_f64; maxl = v if v > maxl }
      sum = 0.0
      probs = Array(Float64).new(ne) do |e|
        p = Math.exp(logits[t, e].to_f64 - maxl)
        sum += p
        p
      end
      probs.map! { |p| p / sum }

      idxs = (0...ne).to_a.sort_by! { |e| -probs[e] }[0, @top_k]
      sel = idxs.map { |e| {e, probs[e]} }
      if norm_topk_prob?
        wsum = sel.sum { |pair| pair[1] }
        wsum = 1.0 if wsum == 0.0
        sel = sel.map { |pair| {pair[0], pair[1] / wsum} }
      end
      sel
    end

    # CPU / quantized path. Each token is routed independently; selected experts
    # run via SwiGLUFF#forward (which itself dispatches to GPU gemv when its
    # weights are quantized), and their outputs are summed with the gate weights.
    def forward(x : SimpleMatrix) : SimpleMatrix
      if result = forward_device_decode(x)
        return result
      end
      if result = forward_prefill_via_device(x)
        return result
      end
      logits = Profile.measure("ffn.router") { router_logits(x) }
      out = SimpleMatrix.zeros(x.rows, @d_model)
      row = SimpleMatrix.new(1, @d_model)
      x.rows.times do |t|
        gating = Profile.measure("ffn.topk") { top_k_gating(logits, t) }
        @d_model.times { |c| row[0, c] = x[t, c] }
        gating.each do |(e, w)|
          ey = @experts[e].forward(row) # [1, d_model]
          Profile.measure("ffn.combine") do
            @d_model.times { |c| out[t, c] = out[t, c] + w * ey[0, c] }
          end
        end
      end
      out
    end

    # Device-resident workspaces for the decode path, allocated once.
    @dev_x : CudaMatrix? = nil
    @dev_x_row : CudaMatrix? = nil
    @dev_out : CudaMatrix? = nil
    @dev_expert_out : CudaMatrix? = nil

    # Rows per batched expert GEMM during prefill. 128 measured best on a 4090 for a
    # 512-token prefill (2.26 ms/tok, against 2.32 at 64 and 2.38 at 32); the curve is
    # flat enough past 64 that this is not worth tuning per model.
    # SHAINET_MOE_TILE overrides it.
    # Rows per batched expert GEMM during prefill. This governs the SHAPE of every expert
    # GEMM, which is where the remaining headroom is: cuBLAS fp32 reaches 4393 GFLOP/s at
    # M=2048 against 792 at M=128, so a bigger M is worth having.
    #
    # 512 measured best on a 4090 at a 4096-token prefill, but only in combination with a
    # prefill chunk large enough to feed it -- the two are one knob in two halves:
    #
    #   chunk 2048 tile  128   33.3 s   8.12 ms/tok   (the old defaults)
    #   chunk 8192 tile  128   33.7 s   8.24          chunk alone changes nothing
    #   chunk 8192 tile  512   29.4 s   7.17          1.13x
    #   chunk 8192 tile 1024   29.4 s   7.17          saturated
    #
    # Chunk alone is flat because tokens-per-expert-per-chunk is chunk * top_k / experts:
    # at chunk 2048 that is 128, so a larger tile has nothing to fill it with.
    # SHAINET_MOE_TILE overrides it.
    #
    # Settable at runtime as well, so a sweep can compare tile sizes within ONE process. A
    # 30B takes about twelve minutes to load, which makes a per-value process prohibitive.
    @@tile_rows : Int32? = nil

    def self.tile_rows : Int32
      v = @@tile_rows
      return v if v
      v = (ENV["SHAINET_MOE_TILE"]? || "512").to_i
      v = 1 if v < 1
      @@tile_rows = v
      v
    end

    # Pass nil to forget the value and re-read the environment.
    def self.tile_rows=(value : Int32?)
      @@tile_rows = value
    end

    # Small device-side scratch for the gather/scatter kernels: the token indices of
    # one expert's slice, and their routing weights. CudaMatrix is Float32 only, so
    # the index side needs its own allocation.
    private class DeviceIndexBuffer
      getter ptr : Pointer(Int32)
      getter capacity : Int32

      def initialize(@capacity : Int32)
        raw = Pointer(Pointer(Void)).malloc(1)
        CUDA.malloc(raw, (@capacity.to_u64 * 4_u64).to_u64)
        @ptr = raw.value.as(Pointer(Int32))
      end

      def upload(host : Array(Int32), n : Int32)
        raise ArgumentError.new("index upload exceeds capacity") if n > @capacity
        CUDA.memcpy(@ptr.as(Pointer(Void)), host.to_unsafe.as(Pointer(Void)),
          (n.to_u64 * 4_u64).to_u64, CUDA::MemcpyKind::HostToDevice)
      end

      def release!
        unless @ptr.null?
          CUDA.free(@ptr.as(Pointer(Void)))
          @ptr = Pointer(Int32).null
        end
      end

      def finalize
        release!
      end
    end

    private class DeviceWeightBuffer
      getter ptr : Pointer(Float32)
      getter capacity : Int32

      def initialize(@capacity : Int32)
        raw = Pointer(Pointer(Void)).malloc(1)
        CUDA.malloc(raw, (@capacity.to_u64 * 4_u64).to_u64)
        @ptr = raw.value.as(Pointer(Float32))
      end

      def upload(host : Array(Float32), n : Int32)
        raise ArgumentError.new("weight upload exceeds capacity") if n > @capacity
        CUDA.memcpy(@ptr.as(Pointer(Void)), host.to_unsafe.as(Pointer(Void)),
          (n.to_u64 * 4_u64).to_u64, CUDA::MemcpyKind::HostToDevice)
      end

      def release!
        unless @ptr.null?
          CUDA.free(@ptr.as(Pointer(Void)))
          @ptr = Pointer(Float32).null
        end
      end

      def finalize
        release!
      end
    end

    @dev_idx : DeviceIndexBuffer? = nil
    @dev_w : DeviceWeightBuffer? = nil

    @@device_decode : Bool? = nil

    # Device-resident decode is the default; SHAINET_MOE_DEVICE=0 forces the host
    # path, which is how the A/B for this change is taken and how the fallback
    # stays exercised.
    def self.device_decode_enabled? : Bool
      flag = @@device_decode
      return flag unless flag.nil?
      flag = ENV.fetch("SHAINET_MOE_DEVICE", "1") != "0"
      @@device_decode = flag
      flag
    end

    # Pass nil to forget the decision and re-read the environment.
    def self.device_decode_enabled=(value : Bool?)
      @@device_decode = value
    end

    # True when the experts can run the device path, so a caller can keep the whole
    # block on the device.
    def device_row_capable? : Bool
      return false unless self.class.device_decode_enabled?
      return false unless CUDA.fully_available?
      return false unless @router.is_a?(CudaMatrix)
      first = @experts.first?
      !!(first && first.device_resident_capable?)
    end

    # Device row in, device row out, with NO readback. This is the variant the block
    # chain uses: the FFN result feeds the block's residual add on the device, so the
    # single readback that forward_device_decode still pays disappears entirely.
    #
    # The returned matrix is a workspace owned by this layer, so consume it before the
    # next call.
    def forward_device_row(x : CudaMatrix) : CudaMatrix
      out_dev = (@dev_out ||= CudaMatrix.new(1, @d_model))
      expert_out = (@dev_expert_out ||= CudaMatrix.new(1, @d_model))

      logits = Profile.measure("ffn.router") { device_router_logits(x) }
      gating = Profile.measure("ffn.topk") { top_k_gating(logits, 0) }

      out_dev.zero!
      handle = CUDA.create_handle
      begin
        gating.each do |(e, w)|
          @experts[e].forward_device(x, expert_out)
          Profile.measure("ffn.dev_combine") do
            CUDA.axpy(handle, w, expert_out.device_ptr.not_nil!, out_dev.device_ptr.not_nil!, @d_model)
          end
        end
      ensure
        CUDA.destroy_handle(handle)
      end
      out_dev.mark_device_dirty!
      out_dev
    end

    # Decode (single token) forward that keeps the activation on the device across
    # every selected expert, so the step costs ONE host-to-device upload and ONE
    # readback instead of three readbacks per expert.
    #
    # Returns nil when the fast path does not apply (prefill, no CUDA, unquantized
    # experts, or a .so without the fused SwiGLU kernel), and the caller falls back
    # to the host path.
    #
    # Routing still round-trips: the top-k choice is a host decision, and the logits
    # are one small row per token rather than per expert, so it is not on the hot
    # path this exists to fix.
    private def forward_device_decode(x : SimpleMatrix) : SimpleMatrix?
      return unless x.rows == 1
      return unless self.class.device_decode_enabled?
      return unless CUDA.fully_available?
      return unless @router.is_a?(CudaMatrix)
      first = @experts.first?
      return unless first && first.device_resident_capable?

      xb = (@dev_x ||= CudaMatrix.new(1, @d_model))
      out_dev = (@dev_out ||= CudaMatrix.new(1, @d_model))
      expert_out = (@dev_expert_out ||= CudaMatrix.new(1, @d_model))

      # One upload, reused by the router GEMM and by every expert.
      Profile.measure("ffn.dev_upload") do
        xb.raw_data.to_unsafe.copy_from(x.data.to_unsafe, @d_model)
        xb.mark_host_modified!
        xb.sync_to_device!("moe_dev_in")
      end

      logits = Profile.measure("ffn.router") { device_router_logits(xb) }
      gating = Profile.measure("ffn.topk") { top_k_gating(logits, 0) }

      out_dev.zero!
      handle = CUDA.create_handle
      begin
        gating.each do |(e, w)|
          @experts[e].forward_device(xb, expert_out)
          # out += w * expert_out, on the device. The GEMV above only launched, so
          # this is ordered behind it on the default stream without a sync.
          Profile.measure("ffn.dev_combine") do
            CUDA.axpy(handle, w, expert_out.device_ptr.not_nil!, out_dev.device_ptr.not_nil!, @d_model)
          end
        end
      ensure
        CUDA.destroy_handle(handle)
      end

      out_dev.mark_device_dirty!
      result = SimpleMatrix.new(1, @d_model)
      # The single readback for the whole FFN, and the only sync point in it.
      Profile.measure("ffn.dev_readback") do
        out_dev.sync_from_device!("moe_dev_out")
        result.data.to_unsafe.copy_from(out_dev.raw_data.to_unsafe, @d_model)
      end
      result
    end

    # fp32 GPU path. Routing + expert evaluation happen through the SimpleMatrix
    # path; convert only at the boundary.
    def forward(x : CudaMatrix) : CudaMatrix
      if result = forward_device_prefill(x)
        return result
      end
      x.sync_from_device!("moe_in") if x.device_dirty?
      forward(x.to_simple).to_cuda
    end

    # Host-facing PREFILL entry. The prefill block chain is SimpleMatrix based
    # (LlamaBlock#forward_cached), so without this the FFN's expert matmuls each
    # bounce to the GPU and back on their own: profiling a 512-token prefill on
    # Qwen3-1.6B-A0.9B measured gemm.out_d2h at 61.7% of the time over 122880 calls
    # (tokens * experts_per_token * 3 matmuls * layers) against 6.7% for the kernels
    # those readbacks wait on.
    #
    # This uploads the activations ONCE, runs every expert on the device, and reads
    # the result back ONCE: two transfers per layer instead of 24 per token per
    # layer. Returns nil when the device path does not apply.
    private def forward_prefill_via_device(x : SimpleMatrix) : SimpleMatrix?
      return unless x.rows > 1
      return unless self.class.device_decode_enabled?
      return unless CUDA.fully_available?
      return unless @router.is_a?(CudaMatrix)
      first = @experts.first?
      return unless first && first.device_resident_capable?

      # Both buffers are SHARED WORKSPACES, not per-call allocations, so neither is
      # freed here: freeing them would hand a dangling pointer to the next layer. They
      # are released by release_prefill_workspaces!. Allocating them per call is what
      # left 134 MB per layer per call to Boehm's finalizers at 16k context.
      dev_in = row_workspace(@@prefill_in, x.rows)
      Profile.measure("ffn.prefill_h2d") do
        dev_in.raw_data.to_unsafe.copy_from(x.data.to_unsafe, x.rows * @d_model)
        dev_in.mark_host_modified!
        dev_in.sync_to_device!("moe_prefill_upload")
      end

      dev_out = forward_device_prefill(dev_in)
      return unless dev_out
      Profile.measure("ffn.prefill_d2h") { dev_out.to_simple }
    end

    # Prefill workspaces, keyed by row count and shared across ALL layers. Two
    # reasons they are class-level rather than per-instance:
    #
    #   - Allocating fresh per call left a [rows, d_model] matrix per layer per call
    #     to Boehm's finalizers. At 16k context that is 134 MB x 48 layers of
    #     allocation churn whose release is not prompt, and VRAM is the binding limit
    #     for context length (measured: 15600 MB of 16376 at 16k).
    #   - Per-instance caching would be worse still: 48 live copies of the same
    #     134 MB buffer.
    #
    # Only one layer runs at a time, so one set is enough. Same reasoning as
    # SwiGLUFF's batch workspaces.
    @@prefill_dst = Hash(Tuple(Int32, Int32), CudaMatrix).new
    @@prefill_in = Hash(Tuple(Int32, Int32), CudaMatrix).new
    @@gathered_shared = Hash(Tuple(Int32, Int32), CudaMatrix).new
    @@batch_out_shared = Hash(Tuple(Int32, Int32), CudaMatrix).new

    # Total VRAM held by the shared prefill workspaces. Exposed so the sharing claim
    # is assertable directly: N layers at one shape must cost ONE set of buffers, not
    # N sets.
    def self.prefill_workspace_bytes : UInt64
      total = 0_u64
      [@@prefill_dst, @@prefill_in, @@gathered_shared, @@batch_out_shared].each do |cache|
        cache.each_value { |m| total += (m.rows.to_u64 * m.cols.to_u64 * 4_u64) }
      end
      total
    end

    # Release every shared prefill workspace. Specs that bound VRAM need this, and so
    # does a caller switching context length: the old shapes are dead weight.
    def self.release_prefill_workspaces! : Nil
      [@@prefill_dst, @@prefill_in, @@gathered_shared, @@batch_out_shared].each do |cache|
        cache.each_value(&.free!)
        cache.clear
      end
    end

    # Public device-in/device-out entry for the multi-token block chain: same batched
    # prefill path the host wrapper uses, without the transfer either side of it.
    def forward_device_batch(x : CudaMatrix) : CudaMatrix?
      forward_device_prefill(x)
    end

    # Device-resident PREFILL: activations stay on the device across every expert.
    # The only transfer is the router logits, one [tokens, num_experts] readback per
    # layer, because top-k is a host decision.
    private def forward_device_prefill(x : CudaMatrix) : CudaMatrix?
      return unless x.rows > 1 # decode has its own tuned path
      return unless self.class.device_decode_enabled?
      return unless CUDA.fully_available?
      return unless CUDA.gather_kernels_available?
      return unless @router.is_a?(CudaMatrix)
      first = @experts.first?
      return unless first && first.device_resident_capable?

      rows = x.rows
      x.sync_to_device!("moe_prefill_in") unless x.device_dirty?

      # One router GEMM for every token, and its logits are the ONLY thing read
      # back: [tokens, num_experts], not per expert.
      logits = Profile.measure("ffn.router") { device_router_logits(x) }

      # Invert the routing: instead of "which experts does this token want", ask
      # "which tokens want this expert". A token's top-k choice is unchanged, so the
      # arithmetic is identical -- only the order the work is issued in differs.
      buckets = Array(Array(Tuple(Int32, Float64))).new(@num_experts) { [] of Tuple(Int32, Float64) }
      Profile.measure("ffn.topk") do
        rows.times do |t|
          top_k_gating(logits, t).each { |(e, w)| buckets[e] << {t, w} }
        end
      end

      dst = row_workspace(@@prefill_dst, rows)
      dst.zero!

      # Tile so the workspaces are a fixed shape, and so one expert with a huge share
      # of the tokens does not size every buffer.
      tile_rows = self.class.tile_rows
      tile = rows < tile_rows ? rows : tile_rows

      # Build the whole layer's plan first, then upload the routing ONCE. Uploading per
      # slice meant a synchronous H2D copy per slice, and each of those drains the
      # pipeline the batched GEMMs are meant to keep full.
      plan = [] of Tuple(Int32, Int32, Int32) # {expert, offset, n}
      all_idx = [] of Int32
      all_w = [] of Float32
      buckets.each_with_index do |toks, e|
        next if toks.empty?
        toks.each_slice(tile) do |slice|
          plan << {e, all_idx.size, slice.size}
          slice.each do |(t, w)|
            all_idx << t
            all_w << w.to_f32
          end
        end
      end
      return dst if plan.empty?

      idx_dev = index_scratch(all_idx.size)
      w_dev = weight_scratch(all_w.size)
      Profile.measure("ffn.plan_upload") do
        idx_dev.upload(all_idx, all_idx.size)
        w_dev.upload(all_w, all_w.size)
      end

      xp = x.device_ptr.not_nil!
      op = dst.device_ptr.not_nil!

      plan.each do |(e, offset, n)|
        # The GEMM is sized to the REAL row count, so a short slice does not pay for a
        # padded tile. Buffers are cached per row count, and the tiling keeps that to a
        # handful of distinct shapes.
        gathered = slice_workspace(@@gathered_shared, n)
        batch_out = slice_workspace(@@batch_out_shared, n)
        idxp = idx_dev.ptr + offset

        Profile.measure("ffn.gather") do
          CUDA.gather_rows(gathered.device_ptr.not_nil!, xp, idxp, n, @d_model)
          gathered.mark_device_dirty!
        end

        # THE point of all of this: three GEMMs for the whole slice, where the
        # per-token path issued three GEMVs per token.
        @experts[e].forward_device_batch(gathered, batch_out)

        Profile.measure("ffn.scatter") do
          CUDA.scatter_add_rows(op, batch_out.device_ptr.not_nil!, idxp,
            w_dev.ptr + offset, n, @d_model)
        end
      end

      dst.mark_device_dirty!
      dst
    end

    # Router logits for device-resident callers. The product is a device temporary and
    # is freed here: left to Boehm it is rows * num_experts * 4 bytes per layer per
    # call, which on a 48-layer model is churn in the same VRAM the expert cache is
    # sized against.
    private def device_router_logits(m : CudaMatrix) : SimpleMatrix
      prod = m * @router.as(CudaMatrix)
      begin
        prod.to_simple
      ensure
        prod.free!
      end
    end

    # Shared, shape-keyed workspaces. The number of distinct shapes MUST be bounded:
    # keying by an exact per-call size is what leaked. Expert routing hands out slice
    # sizes anywhere in 1..tile, so a cache keyed on the exact size accumulates a
    # buffer per distinct size per layer. Summing n = 1..128 at 2048 columns is ~67 MB
    # per cache per layer, and with two caches over 48 layers that reached ~6.5 GB of
    # VRAM that was never released -- observed as an agent whose VRAM climbed
    # 94% -> 100% across two turns and then failed a 19 MB cudaMalloc.
    #
    # Slice buffers are therefore bucketed to the next power of two: at most 8 shapes
    # up to a 128-row tile, and the padding is never worse than 2x. The gather writes
    # only the real rows; the extra rows hold finite leftovers, are computed by the
    # GEMM, and are never scattered back.
    private def bucket(n : Int32) : Int32
      b = 1
      while b < n
        b <<= 1
      end
      b
    end

    private def shared_workspace(cache : Hash(Tuple(Int32, Int32), CudaMatrix), n : Int32) : CudaMatrix
      cache[{n, @d_model}] ||= CudaMatrix.new(n, @d_model)
    end

    private def slice_workspace(cache : Hash(Tuple(Int32, Int32), CudaMatrix), n : Int32) : CudaMatrix
      shared_workspace(cache, bucket(n))
    end

    # Row-count-keyed workspaces cannot be bucketed, because the result is read back
    # at exactly `rows`. Prompt lengths vary per turn, so the cache is CAPPED instead:
    # the least recently used shape is freed. Two entries covers a growing prompt plus
    # the shape a re-prefill lands on.
    MAX_ROW_SHAPES = 2

    private def row_workspace(cache : Hash(Tuple(Int32, Int32), CudaMatrix), rows : Int32) : CudaMatrix
      key = {rows, @d_model}
      if existing = cache[key]?
        # Refresh recency: re-inserting moves it to the end of the iteration order.
        cache.delete(key)
        cache[key] = existing
        return existing
      end

      while cache.size >= MAX_ROW_SHAPES
        oldest = cache.first_key
        cache[oldest].free!
        cache.delete(oldest)
      end
      cache[key] = CudaMatrix.new(rows, @d_model)
    end

    # Grown to fit, never shrunk: the plan is rows * top_k entries, so it settles after
    # the first prefill of a given length.
    private def index_scratch(n : Int32) : DeviceIndexBuffer
      buf = @dev_idx
      if buf.nil? || buf.capacity < n
        buf.try(&.release!)
        buf = @dev_idx = DeviceIndexBuffer.new(n)
      end
      buf
    end

    private def weight_scratch(n : Int32) : DeviceWeightBuffer
      buf = @dev_w
      if buf.nil? || buf.capacity < n
        buf.try(&.release!)
        buf = @dev_w = DeviceWeightBuffer.new(n)
      end
      buf
    end
  end
end
