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
    @dev_out : CudaMatrix? = nil
    @dev_expert_out : CudaMatrix? = nil

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

      logits = Profile.measure("ffn.router") { (xb * @router.as(CudaMatrix)).to_simple }
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
      x.sync_from_device!("moe_in") if x.device_dirty?
      forward(x.to_simple).to_cuda
    end
  end
end
