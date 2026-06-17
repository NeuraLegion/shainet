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

    def initialize(@d_model : Int32, ff_hidden : Int32, @num_experts : Int32,
                   @top_k : Int32, @norm_topk_prob : Bool = true)
      raise ArgumentError.new("num_experts must be positive") unless @num_experts > 0
      raise ArgumentError.new("top_k must be in 1..num_experts") unless 1 <= @top_k <= @num_experts
      @router = SimpleMatrix.new(@d_model, @num_experts)
      # Experts start as empty placeholders; the loader assigns real weights and
      # quantizes them per layer, so we never hold all experts' fp32 at once.
      @experts = Array(SwiGLUFF).new(@num_experts) { SwiGLUFF.new(@d_model, ff_hidden, allocate: false) }
    end

    def to_gpu!(quantize : Bool = false, bits : Int32 = 8)
      return unless CUDA.fully_available?
      # Router stays full precision (CudaMatrix); only experts are quantized.
      @router = @router.as(SimpleMatrix).to_cuda if @router.is_a?(SimpleMatrix)
      @experts.each(&.to_gpu!(quantize, bits))
    end

    # Router logits [tokens, num_experts] for the given activations.
    private def router_logits(x : SimpleMatrix) : SimpleMatrix
      r = @router
      if r.is_a?(CudaMatrix)
        xg = x.to_cuda
        (xg * r).to_simple
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
      logits = router_logits(x)
      out = SimpleMatrix.zeros(x.rows, @d_model)
      row = SimpleMatrix.new(1, @d_model)
      x.rows.times do |t|
        gating = top_k_gating(logits, t)
        @d_model.times { |c| row[0, c] = x[t, c] }
        gating.each do |(e, w)|
          ey = @experts[e].forward(row) # [1, d_model]
          @d_model.times { |c| out[t, c] = out[t, c] + w * ey[0, c] }
        end
      end
      out
    end

    # fp32 GPU path. Routing + expert evaluation happen through the SimpleMatrix
    # path; convert only at the boundary.
    def forward(x : CudaMatrix) : CudaMatrix
      x.sync_from_device!("moe_in") if x.device_dirty?
      forward(x.to_simple).to_cuda
    end
  end
end
