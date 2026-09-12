require "../basic/matrix_layer"

module SHAInet
  # LLaMA-style transformer block with KV cache for efficient generation.
  # Supports Grouped Query Attention (GQA).
  class LlamaBlock < MatrixLayer
    getter norm1 : RMSNorm
    getter norm2 : RMSNorm
    getter ffn : SwiGLUFF | MoEFF
    getter num_heads : Int32
    getter num_kv_heads : Int32
    getter head_dim : Int32
    getter d_model : Int32
    # Total width of the concatenated query heads = num_heads * head_dim. Equals
    # d_model for LLaMA/Qwen2, but Qwen3 uses an explicit head_dim so this can be
    # larger (e.g. 32*128=4096 with d_model=2048). w_q is [d_model, q_dim] and
    # w_o is [q_dim, d_model]; the per-head attention output has width q_dim.
    getter q_dim : Int32
    # Qwen3 QK-norm: per-head RMSNorm applied to Q and K (over head_dim) right
    # before RoPE. nil = disabled (the LLaMA/Qwen2 default — zero overhead).
    property q_norm : Array(Float32)?
    property k_norm : Array(Float32)?
    @qk_norm_eps : Float64 = 1e-6
    property rope_theta : Float64
    # Optional precomputed inverse frequencies (size head_dim/2). When set,
    # these override the default theta^(-2i/d) computation (used for LLaMA 3
    # rope_scaling). nil means use the default.
    property rope_freqs : Array(Float32)?

    property w_q : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_k : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_v : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_o : SimpleMatrix | CudaMatrix | QuantizedWeight

    # Optional Q/K/V projection biases. Qwen2-style architectures add a bias to
    # the query/key/value projections; LLaMA/Mistral do not. Kept as host-side
    # fp32 vectors and added to the projection output before RoPE, so they work
    # identically on the fp32, CUDA, and Q8 weight paths. nil means "no bias"
    # (the LLaMA default — zero overhead, behaviour unchanged). Sizes: b_q is
    # d_model; b_k/b_v are num_kv_heads * head_dim. o_proj has no bias in Qwen2.
    property b_q : Array(Float32)?
    property b_k : Array(Float32)?
    property b_v : Array(Float32)?

    # KV cache: stored per kv_head as [seq_len, head_dim] growing matrices
    @k_cache : Array(Array(Float32)) # [num_kv_heads][seq_len * head_dim]
    @v_cache : Array(Array(Float32))
    @cache_len : Int32 = 0

    # GPU-resident mirror of the KV cache, laid out [num_kv_heads, capacity,
    # head_dim] per tensor. The CPU cache stays the source of truth: the GPU
    # copy is appended incrementally each token and fully re-uploaded from the
    # mirror whenever capacity grows. Only used when CUDA kernels are loaded.
    #
    # Stored as an untyped pointer because the element type is chosen at first
    # allocation: fp16 halves the cache's VRAM footprint, and at long context
    # the cache is the single largest consumer (0.375 MB per token for a 48-layer
    # 8-KV-head model in fp32). Arithmetic stays fp32 in both cases — the kernels
    # convert on load — so only the stored precision differs.
    @gpu_k_cache : Pointer(Void) = Pointer(Void).null
    @gpu_v_cache : Pointer(Void) = Pointer(Void).null
    @gpu_cache_cap : Int32 = 0
    # nil until the first device allocation decides the cache dtype.
    @kv_fp16 : Bool? = nil
    # Context budget: when set the device cache is allocated once at exactly this
    # many positions rather than doubling into it. Resolved from
    # SHAINET_KV_MAX_CONTEXT on first read; the checked flag keeps a deliberate
    # nil from re-reading the environment on every forward pass.
    @kv_max_context : Int32? = nil
    @kv_max_context_checked : Bool = false
    # Persistent device scratch for the attention hot path. SHARED across all
    # blocks (class-level): blocks run sequentially, so one grow-only scratch set
    # serves every layer instead of each of N layers holding its own ~(heads×
    # seq×seq) workspace (which was ~N× redundant and exhausted VRAM on long
    # prefills). The KV cache above stays per-instance.
    @@gpu_staging : Pointer(Float32) = Pointer(Float32).null
    @@gpu_staging_cap : Int32 = 0
    @@gpu_attn_out : Pointer(Float32) = Pointer(Float32).null
    @@gpu_attn_out_cap : Int32 = 0
    @@gpu_attn_ws : Pointer(Float32) = Pointer(Float32).null
    @@gpu_attn_ws_cap : Int32 = 0
    @@staging_host : Array(Float32) = Array(Float32).new
    @gpu_attn_avail : Bool?
    # Force the CPU attention path even when CUDA is available (tests/fallback).
    property? force_cpu_attention : Bool = false

    # Ceiling on the attention workspace, in fp32 elements (256 MB). Prefill is
    # chunked so num_heads * chunk_tokens * total_len stays under this; without
    # chunking the workspace grows as num_heads * seq^2, which is the single
    # sharpest VRAM cliff on a long prompt.
    ATTN_WS_BUDGET_FLOATS = 64_i64 * 1024 * 1024
    # Upper bound on chunk size regardless of budget. Larger chunks stop paying
    # off once the per-launch work saturates the device.
    ATTN_CHUNK_MAX = 256

    def initialize(@d_model : Int32, @num_heads : Int32, ff_hidden : Int32,
                   eps : Float64 = 1e-6, @rope_theta : Float64 = 10000.0,
                   @num_kv_heads : Int32 = @num_heads, head_dim : Int32? = nil,
                   moe_experts : Int32? = nil, moe_top_k : Int32 = 8,
                   moe_norm_topk : Bool = true, moe_ff_hidden : Int32? = nil,
                   moe_offload : Bool = false)
      super(@d_model, SHAInet.none)
      raise ArgumentError.new("num_heads must be divisible by num_kv_heads") unless @num_kv_heads > 0 && @num_heads % @num_kv_heads == 0
      # head_dim defaults to d_model/num_heads (LLaMA/Qwen2). Qwen3 passes it
      # explicitly (e.g. 128), so q_dim = num_heads*head_dim may differ from d_model.
      if hd = head_dim
        raise ArgumentError.new("head_dim must be positive") unless hd > 0
        @head_dim = hd
      else
        raise ArgumentError.new("d_model must be divisible by num_heads") unless @d_model % @num_heads == 0
        @head_dim = @d_model // @num_heads
      end
      @q_dim = @num_heads * @head_dim
      kv_dim = @num_kv_heads * @head_dim
      @qk_norm_eps = eps
      @norm1 = RMSNorm.new(@d_model, eps)
      @norm2 = RMSNorm.new(@d_model, eps)
      # Mixture-of-Experts FFN (Qwen3-MoE) when moe_experts is given; otherwise a
      # single dense SwiGLU (LLaMA/Mistral/Qwen2/Qwen3-dense).
      @ffn = if ne = moe_experts
               MoEFF.new(@d_model, moe_ff_hidden || ff_hidden, ne, moe_top_k, moe_norm_topk, moe_offload)
             else
               SwiGLUFF.new(@d_model, ff_hidden)
             end
      @w_q = SimpleMatrix.new(@d_model, @q_dim)
      @w_k = SimpleMatrix.new(@d_model, kv_dim)
      @w_v = SimpleMatrix.new(@d_model, kv_dim)
      @w_o = SimpleMatrix.new(@q_dim, @d_model)
      @k_cache = Array.new(@num_kv_heads) { Array(Float32).new }
      @v_cache = Array.new(@num_kv_heads) { Array(Float32).new }
    end

    # Persistent single-row GEMV workspaces for decode (M=1), keyed by width.
    # Reused across tokens to avoid per-call cudaMalloc/cudaFree churn. Never
    # freed during inference, so they cannot be GC-collected mid-GEMM.
    @q8_in_bufs = Hash(Int32, CudaMatrix).new
    @q8_out_bufs = Hash(Int32, CudaMatrix).new

    def clear_cache!
      @k_cache.each(&.clear)
      @v_cache.each(&.clear)
      @cache_len = 0
      # Device cache buffers are kept; stale positions are rewritten by the
      # append kernel before they ever become visible to attention.
    end

    def finalize
      # Only free the per-instance KV cache; the attention scratch is class-level
      # and shared across all blocks, so it must not be freed per instance.
      {@gpu_k_cache, @gpu_v_cache}.each do |p|
        CUDA.free(p) unless p.null?
      end
    end

    def to_gpu!(quantize : Bool = false, bits : Int32 = 8)
      return unless CUDA.fully_available?
      if quantize
        @w_q = to_quant(@w_q, bits)
        @w_k = to_quant(@w_k, bits)
        @w_v = to_quant(@w_v, bits)
        @w_o = to_quant(@w_o, bits)
      else
        # Only promote host weights; leave existing CudaMatrix/QuantizedWeight as-is.
        @w_q = @w_q.as(SimpleMatrix).to_cuda if @w_q.is_a?(SimpleMatrix)
        @w_k = @w_k.as(SimpleMatrix).to_cuda if @w_k.is_a?(SimpleMatrix)
        @w_v = @w_v.as(SimpleMatrix).to_cuda if @w_v.is_a?(SimpleMatrix)
        @w_o = @w_o.as(SimpleMatrix).to_cuda if @w_o.is_a?(SimpleMatrix)
      end
      @norm1.to_gpu!
      @norm2.to_gpu!
      @ffn.to_gpu!(quantize, bits)
    end

    # Quantize a weight to the requested bit width: bits == 4 -> Q4, bits == 8 ->
    # Q8. Already-quantized weights are returned unchanged (we quantize once
    # during load, so the format is never switched in place).
    private def to_quant(w : SimpleMatrix | CudaMatrix | QuantizedWeight, bits : Int32) : QuantizedWeight
      raise ArgumentError.new("unsupported quantization bits: #{bits} (expected 8 or 4)") unless bits == 8 || bits == 4
      case w
      when QuantizedWeight then w
      when CudaMatrix
        sm = w.to_simple
        bits == 4 ? Q4CudaMatrix.from_simple(sm) : QuantizedCudaMatrix.from_simple(sm)
      else
        sm = w.as(SimpleMatrix)
        bits == 4 ? Q4CudaMatrix.from_simple(sm) : QuantizedCudaMatrix.from_simple(sm)
      end
    end

    def apply_gradients(lr : Float64)
    end

    def backward(d_out : SimpleMatrix) : SimpleMatrix
      raise "LlamaBlock backward pass not yet implemented"
    end

    def backward(d_out : CudaMatrix) : CudaMatrix
      raise "LlamaBlock backward pass not yet implemented"
    end

    # CPU forward — full sequence (no cache, for prefill or training)
    def forward(x : SimpleMatrix) : SimpleMatrix
      normed = @norm1.forward(x)
      attn = attention_full_cpu(normed)
      h = x + attn
      normed2 = @norm2.forward(h)
      ff_out = @ffn.forward(normed2)
      h + ff_out
    end

    # CPU forward with KV cache — only processes new tokens
    def forward_cached(x : SimpleMatrix) : SimpleMatrix
      normed = Profile.measure("block.norm") { @norm1.forward(x) }
      attn = attention_cached_cpu(normed)
      h = Profile.measure("block.residual") { x + attn }
      normed2 = Profile.measure("block.norm") { @norm2.forward(h) }
      ff_out = @ffn.forward(normed2)
      Profile.measure("block.residual") { h + ff_out }
    end

    # Device-resident workspaces for the block chain, one set per block.
    @dev_n1 : CudaMatrix? = nil
    @dev_n2 : CudaMatrix? = nil
    @dev_attn : CudaMatrix? = nil
    @dev_h : CudaMatrix? = nil

    @@block_device : Bool? = nil

    # Device-resident block chain is the default; SHAINET_BLOCK_DEVICE=0 forces the
    # host path, which is how the A/B is taken and how the fallback stays exercised.
    def self.block_device_enabled? : Bool
      flag = @@block_device
      return flag unless flag.nil?
      flag = ENV.fetch("SHAINET_BLOCK_DEVICE", "1") != "0"
      @@block_device = flag
      flag
    end

    # Pass nil to forget the decision and re-read the environment.
    def self.block_device_enabled=(value : Bool?)
      @@block_device = value
    end

    # True when the whole block can run with the activation staying on the device:
    # both norms have their kernel and gamma there, the attention projections and
    # o_proj are quantized, and the FFN has a device path.
    def block_device_capable? : Bool
      return false unless self.class.block_device_enabled?
      return false unless CUDA.fully_available? && CUDA.block_device_kernels_available?
      return false unless @norm1.device_capable? && @norm2.device_capable?
      return false unless @w_q.is_a?(QuantizedWeight) && @w_k.is_a?(QuantizedWeight) &&
                          @w_v.is_a?(QuantizedWeight) && @w_o.is_a?(QuantizedWeight)
      return false unless gpu_attention?
      ffn = @ffn
      ffn.is_a?(MoEFF) ? ffn.device_row_capable? : false
    end

    # Decode forward for a single token whose activation never leaves the device.
    #
    # The host path spends, per layer: two host RMSNorm loops, two host residual adds,
    # three uploads of the same normed row for q/k/v, a readback of the o_proj result
    # and a readback of the FFN result. All of those are gone here. What remains is
    # inherent to the attention implementation: q/k/v come back because RoPE, the KV
    # host mirror and the attention staging are host-side, so this does NOT claim to
    # make attention device-resident.
    #
    # `x` must be device-resident; the returned matrix is this block's own workspace,
    # so the caller must consume or hand it straight to the next block.
    def forward_cached_device(x : CudaMatrix) : CudaMatrix
      dm = @d_model
      n1 = (@dev_n1 ||= CudaMatrix.new(1, dm))
      n2 = (@dev_n2 ||= CudaMatrix.new(1, dm))
      attn_dev = (@dev_attn ||= CudaMatrix.new(1, dm))
      h = (@dev_h ||= CudaMatrix.new(1, dm))

      Profile.measure("block.dev_norm") { @norm1.forward_into(x, n1) }
      attention_cached_device(n1, attn_dev)

      # h = x + attn, then h += ffn(norm2(h)), both residuals on the device.
      Profile.measure("block.dev_residual") do
        CUDA.memcpy(h.device_ptr.not_nil!.as(Pointer(Void)), x.device_ptr.not_nil!.as(Pointer(Void)),
          dm.to_u64 * 4_u64, CUDA::MemcpyKind::DeviceToDevice)
        CUDA.add_inplace(h.device_ptr.not_nil!, attn_dev.device_ptr.not_nil!, dm)
        h.mark_device_dirty!
      end

      Profile.measure("block.dev_norm") { @norm2.forward_into(h, n2) }
      ff_out = @ffn.as(MoEFF).forward_device_row(n2)
      Profile.measure("block.dev_residual") do
        CUDA.add_inplace(h.device_ptr.not_nil!, ff_out.device_ptr.not_nil!, dm)
        h.mark_device_dirty!
      end
      h
    end

    # Attention for the device chain: device row in, device row out. The interior is
    # unchanged from the host path, so RoPE/KV/staging behaviour and the fp16 cache
    # are identical; only the boundaries move.
    private def attention_cached_device(x : CudaMatrix, dst : CudaMatrix) : Nil
      new_tokens = 1
      head_dim = @head_dim
      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
      start_pos = @cache_len

      q_full = gemv_from_device(x, @w_q.as(QuantizedWeight))
      k_new = gemv_from_device(x, @w_k.as(QuantizedWeight))
      v_new = gemv_from_device(x, @w_v.as(QuantizedWeight))
      add_bias!(q_full, @b_q)
      add_bias!(k_new, @b_k)
      add_bias!(v_new, @b_v)
      apply_head_rmsnorm!(q_full, @num_heads, @q_norm)
      apply_head_rmsnorm!(k_new, @num_kv_heads, @k_norm)

      half = head_dim // 2
      Profile.measure("attn.rope_kv_append") do
        @num_kv_heads.times do |kv_h|
          kv_col = kv_h * head_dim
          pos = start_pos
          rotated = Array(Float32).new(head_dim, 0.0_f32)
          half.times do |i|
            angle = (pos * inv_freq(i)).to_f32
            cos_val = Math.cos(angle).to_f32
            sin_val = Math.sin(angle).to_f32
            x0 = k_new[0, kv_col + i].to_f32
            x1 = k_new[0, kv_col + i + half].to_f32
            rotated[i] = x0 * cos_val - x1 * sin_val
            rotated[i + half] = x1 * cos_val + x0 * sin_val
          end
          rotated.each { |val| @k_cache[kv_h] << val }
          head_dim.times { |d| @v_cache[kv_h] << v_new[0, kv_col + d].to_f32 }
        end
      end

      total_len = @cache_len + new_tokens
      @cache_len = total_len
      output = SimpleMatrix.new(new_tokens, @q_dim)
      attention_heads_gpu(q_full, output, new_tokens, start_pos, total_len, scale)

      gemv_to_device(output, @w_o.as(QuantizedWeight), dst)
    end

    # GPU forward — full sequence
    def forward(x : CudaMatrix) : CudaMatrix
      normed = @norm1.forward(x)
      attn = attention_full_gpu(normed)
      h = x + attn
      normed2 = @norm2.forward(h)
      ff_out = @ffn.forward(normed2)
      h + ff_out
    end

    # --- CPU attention with full recompute ---
    private def attention_full_cpu(x : SimpleMatrix) : SimpleMatrix
      seq_len = x.rows
      head_dim = @head_dim
      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32

      q_full = gpu_matmul(x, @w_q) # [seq, d_model]
      k_full = gpu_matmul(x, @w_k) # [seq, kv_dim]
      v_full = gpu_matmul(x, @w_v) # [seq, kv_dim]
      add_bias!(q_full, @b_q)
      add_bias!(k_full, @b_k)
      add_bias!(v_full, @b_v)
      apply_head_rmsnorm!(q_full, @num_heads, @q_norm)
      apply_head_rmsnorm!(k_full, @num_kv_heads, @k_norm)

      output = SimpleMatrix.new(seq_len, @q_dim)
      heads_per_kv = @num_heads // @num_kv_heads

      @num_heads.times do |h|
        q_col = h * head_dim
        kv_h = h // heads_per_kv
        kv_col = kv_h * head_dim

        # Extract + RoPE
        q_h = extract_head(q_full, seq_len, q_col, head_dim)
        k_h = extract_head(k_full, seq_len, kv_col, head_dim)
        v_h = extract_head(v_full, seq_len, kv_col, head_dim)
        apply_rope!(q_h, 0)
        apply_rope!(k_h, 0)

        # Attention: Q * K^T * scale, causal mask, softmax, * V
        causal_attention!(output, q_h, k_h, v_h, q_col, scale)
      end

      gpu_matmul(output, @w_o)
    end

    # --- CPU attention with KV cache (incremental) ---
    private def attention_cached_cpu(x : SimpleMatrix) : SimpleMatrix
      new_tokens = x.rows
      head_dim = @head_dim
      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
      start_pos = @cache_len

      # Project Q/K/V — use GPU GEMM if weights are on device
      q_full = gpu_matmul(x, @w_q)
      k_new = gpu_matmul(x, @w_k)
      v_new = gpu_matmul(x, @w_v)
      # Qwen2-style projection biases (no-op when unset, i.e. LLaMA).
      add_bias!(q_full, @b_q)
      add_bias!(k_new, @b_k)
      add_bias!(v_new, @b_v)
      # Qwen3 QK-norm before RoPE (no-op when unset). K is normalized here, then
      # rotated and appended below; Q is normalized here, then rotated in the
      # per-head attention loop.
      apply_head_rmsnorm!(q_full, @num_heads, @q_norm)
      apply_head_rmsnorm!(k_new, @num_kv_heads, @k_norm)

      # Apply RoPE to new K at insert time (HF half-split), then append to cache.
      half = head_dim // 2
      Profile.measure("attn.rope_kv_append") do
        @num_kv_heads.times do |kv_h|
          kv_col = kv_h * head_dim
          new_tokens.times do |t|
            pos = start_pos + t
            rotated = Array(Float32).new(head_dim, 0.0_f32)
            half.times do |i|
              freq = inv_freq(i)
              angle = (pos * freq).to_f32
              cos_val = Math.cos(angle).to_f32
              sin_val = Math.sin(angle).to_f32
              x0 = k_new[t, kv_col + i].to_f32
              x1 = k_new[t, kv_col + i + half].to_f32
              rotated[i] = x0 * cos_val - x1 * sin_val
              rotated[i + half] = x1 * cos_val + x0 * sin_val
            end
            rotated.each { |val| @k_cache[kv_h] << val }
            head_dim.times { |d| @v_cache[kv_h] << v_new[t, kv_col + d].to_f32 }
          end
        end
      end

      total_len = @cache_len + new_tokens
      @cache_len = total_len
      output = SimpleMatrix.new(new_tokens, @q_dim)

      if gpu_attention?
        attention_heads_gpu(q_full, output, new_tokens, start_pos, total_len, scale)
      else
        attention_heads_cpu(q_full, output, new_tokens, start_pos, total_len, scale)
      end

      gpu_matmul(output, @w_o)
    end

    # --- CPU head loop: scores -> softmax -> AV, per query head/token ---
    private def attention_heads_cpu(q_full : SimpleMatrix, output : SimpleMatrix,
                                    new_tokens : Int32, start_pos : Int32,
                                    total_len : Int32, scale : Float32)
      head_dim = @head_dim
      heads_per_kv = @num_heads // @num_kv_heads

      half = head_dim // 2
      dm = @q_dim
      qptr = q_full.data.to_unsafe
      optr = output.data.to_unsafe
      # Reusable scratch buffers (avoid per-head/per-token allocations).
      q_rot = Array(Float32).new(head_dim, 0.0_f32)
      scores = Array(Float32).new(total_len, 0.0_f32)
      out = Array(Float32).new(head_dim, 0.0_f32)
      # Raw pointers bypass Array bounds-checks in the hot inner loops and let
      # the compiler vectorize the dot-product / weighted-sum reductions.
      qrp = q_rot.to_unsafe
      scp = scores.to_unsafe
      outp = out.to_unsafe

      @num_heads.times do |h|
        q_col = h * head_dim
        kv_h = h // heads_per_kv
        kptr = @k_cache[kv_h].to_unsafe
        vptr = @v_cache[kv_h].to_unsafe

        new_tokens.times do |i|
          pos = start_pos + i
          qrow = i * dm + q_col

          # RoPE-rotate this token's Q head directly into q_rot (HF half-split).
          idx = 0
          while idx < half
            freq = inv_freq(idx)
            angle = (pos * freq).to_f32
            c = Math.cos(angle).to_f32
            s = Math.sin(angle).to_f32
            x0 = qptr[qrow + idx]
            x1 = qptr[qrow + idx + half]
            qrp[idx] = x0 * c - x1 * s
            qrp[idx + half] = x1 * c + x0 * s
            idx += 1
          end

          # scores[j] = scale * (q_rot · K_cache[j]); track max for stable softmax.
          visible = start_pos + i + 1
          max_val = -Float32::INFINITY
          j = 0
          while j < visible
            kbase = j * head_dim
            dot = 0.0_f32
            d = 0
            while d < head_dim
              dot += qrp[d] * kptr[kbase + d]
              d += 1
            end
            sv = dot * scale
            scp[j] = sv
            max_val = sv if sv > max_val
            j += 1
          end

          # softmax over visible positions
          exp_sum = 0.0_f32
          j = 0
          while j < visible
            e = Math.exp((scp[j] - max_val).to_f64).to_f32
            scp[j] = e
            exp_sum += e
            j += 1
          end
          inv_sum = 1.0_f32 / exp_sum

          # out = sum_j (softmax_j) * V_cache[j]
          d = 0
          while d < head_dim
            outp[d] = 0.0_f32
            d += 1
          end
          j = 0
          while j < visible
            w = scp[j] * inv_sum
            vbase = j * head_dim
            d = 0
            while d < head_dim
              outp[d] += w * vptr[vbase + d]
              d += 1
            end
            j += 1
          end

          orow = i * dm + q_col
          d = 0
          while d < head_dim
            optr[orow + d] = outp[d]
            d += 1
          end
        end
      end
    end

    # --- GPU head loop: fused scores/softmax/AV over the device KV cache ---
    # One staging upload carries the new K/V rows plus the RoPE'd Q, then two
    # kernels (cache append + attention) run on-device and only the attention
    # output [new_tokens, d_model] is copied back.
    private def attention_heads_gpu(q_full : SimpleMatrix, output : SimpleMatrix,
                                    new_tokens : Int32, start_pos : Int32,
                                    total_len : Int32, scale : Float32)
      head_dim = @head_dim
      dm = @q_dim
      half = head_dim // 2
      heads_per_kv = @num_heads // @num_kv_heads

      # Prefill is processed in bounded chunks of query tokens. The attention
      # workspace is O(num_heads * tokens * total_len), so attending a whole
      # long prompt in one launch is quadratic in the prompt length (a 32k
      # prefill with 32 heads would need ~200 GB of scratch). Chunking caps the
      # workspace at ATTN_WS_BUDGET_FLOATS and also bounds the staging and
      # output buffers, which were previously linear in the full prompt length.
      max_chunk = attn_chunk_tokens(total_len)
      max_chunk = new_tokens if new_tokens < max_chunk

      chunk_floats = max_chunk * head_dim   # floats per kv_head per tensor
      kv_cap = @num_kv_heads * chunk_floats # staging size of K (and of V)
      q_cap = max_chunk * dm
      staging_floats = 2 * kv_cap + q_cap
      ws_floats = @num_heads * max_chunk * total_len

      @@gpu_staging, @@gpu_staging_cap = grow_dev_buf(@@gpu_staging, @@gpu_staging_cap, staging_floats)
      @@gpu_attn_out, @@gpu_attn_out_cap = grow_dev_buf(@@gpu_attn_out, @@gpu_attn_out_cap, q_cap)
      @@gpu_attn_ws, @@gpu_attn_ws_cap = grow_dev_buf(@@gpu_attn_ws, @@gpu_attn_ws_cap, ws_floats,
        cap_limit: ATTN_WS_BUDGET_FLOATS.to_i32)

      st = @@staging_host
      if st.size < staging_floats
        st = Array(Float32).new(staging_floats, 0.0_f32)
        @@staging_host = st
      end
      stp = st.to_unsafe

      # Allocated last: growing the cache re-uploads the host mirror through the
      # append kernel, which needs the staging buffers above already sized.
      ensure_gpu_cache!(total_len, max_chunk)

      qptr = q_full.data.to_unsafe
      outp = output.data.to_unsafe
      # cos/sin depend only on (pos, rotation index), so compute once per token.
      cosv = Array(Float32).new(half, 0.0_f32)
      sinv = Array(Float32).new(half, 0.0_f32)
      cp = cosv.to_unsafe
      sp = sinv.to_unsafe

      off = 0
      while off < new_tokens
        n = Math.min(max_chunk, new_tokens - off)
        base_pos = start_pos + off
        cf = n * head_dim
        kvf = @num_kv_heads * cf
        qf = n * dm

        Profile.measure("attn.stage_host") do
          # This chunk's new K/V rows: each kv_head's slice is contiguous in the
          # CPU mirror (RoPE already applied to K at insert).
          tail = base_pos * head_dim
          @num_kv_heads.times do |kv_h|
            (stp + kv_h * cf).copy_from(@k_cache[kv_h].to_unsafe + tail, cf)
            (stp + kvf + kv_h * cf).copy_from(@v_cache[kv_h].to_unsafe + tail, cf)
          end

          # RoPE-rotate this chunk's Q (HF half-split) into the staging blob,
          # token-major and re-based to the chunk's own first row.
          qst = stp + 2 * kvf
          n.times do |i|
            pos = base_pos + i
            half.times do |r|
              angle = (pos * inv_freq(r)).to_f32
              cp[r] = Math.cos(angle).to_f32
              sp[r] = Math.sin(angle).to_f32
            end
            src_row = (off + i) * dm
            dst_row = i * dm
            @num_heads.times do |h|
              sb = src_row + h * head_dim
              db = dst_row + h * head_dim
              r = 0
              while r < half
                x0 = qptr[sb + r]
                x1 = qptr[sb + r + half]
                qst[db + r] = x0 * cp[r] - x1 * sp[r]
                qst[db + r + half] = x1 * cp[r] + x0 * sp[r]
                r += 1
              end
            end
          end
        end

        Profile.measure("attn.h2d") do
          CUDA.memcpy(@@gpu_staging.as(Pointer(Void)), stp.as(Pointer(Void)),
            (2 * kvf + qf).to_u64 * 4_u64, CUDA::MemcpyKind::HostToDevice)
        end
        Profile.measure("attn.kernels") do
          append_kv(n, base_pos)
          attend_kv(@@gpu_staging + 2 * kvf, n, base_pos, heads_per_kv, scale)
        end
        # Synchronous D2H read-back also orders after both kernels above.
        Profile.measure("attn.d2h") do
          CUDA.memcpy((outp + off * dm).as(Pointer(Void)), @@gpu_attn_out.as(Pointer(Void)),
            qf.to_u64 * 4_u64, CUDA::MemcpyKind::DeviceToHost)
        end

        off += n
      end
    end

    # Query tokens attended per kernel launch. Sized so the attention scratch
    # (num_heads * tokens * total_len floats) stays within ATTN_WS_BUDGET_FLOATS,
    # capped at ATTN_CHUNK_MAX. Override with SHAINET_ATTN_CHUNK.
    #
    # Public so the budget math can be asserted directly rather than inferred
    # from an allocation side effect.
    def attn_chunk_tokens(total_len : Int32) : Int32
      if env = ENV["SHAINET_ATTN_CHUNK"]?
        forced = env.to_i?
        return forced if forced && forced > 0
      end
      per_token = @num_heads.to_i64 * total_len.to_i64
      return ATTN_CHUNK_MAX if per_token <= 0
      n = (ATTN_WS_BUDGET_FLOATS // per_token).to_i32
      return 1 if n < 1
      n > ATTN_CHUNK_MAX ? ATTN_CHUNK_MAX : n
    end

    # GPU attention is used whenever the CUDA kernels are loadable and the
    # caller has not forced the CPU path (property or SHAINET_CPU_ATTENTION=1).
    # Memoized: fully_available? dlopens.
    private def gpu_attention? : Bool
      return false if force_cpu_attention?
      avail = @gpu_attn_avail
      if avail.nil?
        avail = !ENV["SHAINET_CPU_ATTENTION"]? && CUDA.fully_available?
        @gpu_attn_avail = avail
      end
      avail
    end

    # Whether the device KV cache is kept in fp16. Decided once, at the first
    # allocation, and then fixed for the lifetime of the buffers so a resize
    # never has to reinterpret existing contents. Requires the fp16 kernels to
    # be present in the loaded kernel library (older prebuilt .so files predate
    # them); opt out with SHAINET_KV_FP16=0.
    def kv_cache_fp16? : Bool
      flag = @kv_fp16
      return flag unless flag.nil?
      flag = ENV.fetch("SHAINET_KV_FP16", "1") != "0" && CUDA.kv_f16_kernels_available?
      @kv_fp16 = flag
      flag
    end

    # Current device footprint of this block's KV cache in bytes (K and V), at
    # the allocated capacity rather than the used length.
    def kv_cache_bytes : UInt64
      return 0_u64 if @gpu_cache_cap == 0
      2_u64 * @num_kv_heads.to_u64 * @gpu_cache_cap.to_u64 * @head_dim.to_u64 * kv_elem_bytes
    end

    # Bytes per cache element for the active dtype.
    private def kv_elem_bytes : UInt64
      kv_cache_fp16? ? 2_u64 : 4_u64
    end

    # Ensure the device KV cache holds at least total_len positions per
    # kv_head.
    #
    # With a context budget configured (see #kv_max_context) the cache is
    # allocated ONCE at exactly that size. Without one it grows by doubling,
    # which overshoots badly: a measured 4128-token context allocated 8192 slots,
    # so half the cache was never used.
    #
    # On (re)allocation the CPU mirror, which already contains the new tokens, is
    # re-uploaded into the fresh buffers. `max_chunk` bounds the staging used by
    # that re-upload, so this must be called after the staging buffers have been
    # sized for that chunk.
    private def ensure_gpu_cache!(total_len : Int32, max_chunk : Int32)
      return if @gpu_cache_cap >= total_len
      head_dim = @head_dim

      new_cap = if budget = kv_max_context
                  if total_len > budget
                    raise ArgumentError.new(
                      "context length #{total_len} exceeds the configured KV budget #{budget}; " \
                      "raise kv_max_context (or SHAINET_KV_MAX_CONTEXT) or shorten the prompt")
                  end
                  budget
                else
                  Math.max(256, Math.max(total_len, @gpu_cache_cap * 2))
                end

      bytes = @num_kv_heads.to_u64 * new_cap.to_u64 * head_dim.to_u64 * kv_elem_bytes
      CUDA.free(@gpu_k_cache) unless @gpu_k_cache.null?
      CUDA.free(@gpu_v_cache) unless @gpu_v_cache.null?
      kp = Pointer(Void).null
      vp = Pointer(Void).null
      CUDA.malloc(pointerof(kp), bytes)
      CUDA.malloc(pointerof(vp), bytes)
      @gpu_k_cache = kp
      @gpu_v_cache = vp
      @gpu_cache_cap = new_cap

      used = @num_kv_heads > 0 ? @k_cache[0].size // head_dim : 0
      reupload_gpu_cache!(used, max_chunk) if used > 0
    end

    # Upper bound on cached positions. When set, the device KV cache is sized to
    # exactly this once instead of doubling into it, which removes the overshoot
    # (up to 2x) that doubling leaves behind, and removes the repeated
    # free/malloc/re-upload cycle on the way there. The host mirror reserves the
    # same capacity so it stops reallocating as it grows.
    #
    # Defaults from SHAINET_KV_MAX_CONTEXT; nil keeps the doubling behaviour.
    def kv_max_context : Int32?
      cached = @kv_max_context
      return cached if cached
      return if @kv_max_context_checked
      @kv_max_context_checked = true
      if raw = ENV["SHAINET_KV_MAX_CONTEXT"]?
        if v = raw.to_i?
          @kv_max_context = v if v > 0
        end
      end
      @kv_max_context
    end

    # Set the context budget explicitly. Reserves host mirror capacity right
    # away; the device buffers are sized on the next forward pass. Raises if the
    # device cache has already grown past the new budget, since shrinking it
    # would discard cached positions the caller may still be attending to.
    def kv_max_context=(value : Int32?)
      if v = value
        raise ArgumentError.new("kv_max_context must be positive") unless v > 0
        if @gpu_cache_cap > v
          raise ArgumentError.new(
            "device KV cache is already #{@gpu_cache_cap} positions; " \
            "clear_cache! and free it before lowering the budget to #{v}")
        end
        reserve_host_cache!(v)
      end
      @kv_max_context_checked = true
      @kv_max_context = value
    end

    # Grow the host mirror's capacity without changing its length, so appending
    # up to `positions` tokens does not reallocate mid-generation.
    private def reserve_host_cache!(positions : Int32)
      want = positions.to_i64 * @head_dim
      return if want <= 0
      @num_kv_heads.times do |kv_h|
        {@k_cache, @v_cache}.each do |cache|
          arr = cache[kv_h]
          next if arr.size >= want
          grown = Array(Float32).new(want)
          grown.concat(arr)
          cache[kv_h] = grown
        end
      end
    end

    # Re-populate the freshly grown device cache from the host mirror, in chunks
    # of at most max_chunk positions. This goes through the append kernel rather
    # than a raw memcpy so the fp32 mirror is converted for an fp16 cache, and so
    # exactly one layout description exists for both dtypes.
    private def reupload_gpu_cache!(upto : Int32, max_chunk : Int32)
      head_dim = @head_dim
      step = max_chunk < 1 ? 1 : max_chunk
      st = @@staging_host
      stp = st.to_unsafe

      pos = 0
      while pos < upto
        n = Math.min(step, upto - pos)
        cf = n * head_dim
        kvf = @num_kv_heads * cf
        src_off = pos * head_dim
        @num_kv_heads.times do |kv_h|
          (stp + kv_h * cf).copy_from(@k_cache[kv_h].to_unsafe + src_off, cf)
          (stp + kvf + kv_h * cf).copy_from(@v_cache[kv_h].to_unsafe + src_off, cf)
        end
        CUDA.memcpy(@@gpu_staging.as(Pointer(Void)), stp.as(Pointer(Void)),
          (2 * kvf).to_u64 * 4_u64, CUDA::MemcpyKind::HostToDevice)
        append_kv(n, pos)
        pos += n
      end
    end

    # Dispatch the cache-append kernel for the active cache dtype. Staging is
    # always fp32 and lives at the head of @@gpu_staging.
    private def append_kv(n : Int32, base_pos : Int32)
      if kv_cache_fp16?
        CUDA.kv_cache_append_f16(@@gpu_staging, @gpu_k_cache.as(Pointer(UInt16)),
          @gpu_v_cache.as(Pointer(UInt16)), n, base_pos, @num_kv_heads,
          @head_dim, @gpu_cache_cap)
      else
        CUDA.kv_cache_append_f32(@@gpu_staging, @gpu_k_cache.as(Pointer(Float32)),
          @gpu_v_cache.as(Pointer(Float32)), n, base_pos, @num_kv_heads,
          @head_dim, @gpu_cache_cap)
      end
    end

    # Dispatch the attention kernel for the active cache dtype.
    private def attend_kv(q : Pointer(Float32), n : Int32, base_pos : Int32,
                          heads_per_kv : Int32, scale : Float32)
      if kv_cache_fp16?
        CUDA.attention_kv_f16(q, @gpu_k_cache.as(Pointer(UInt16)),
          @gpu_v_cache.as(Pointer(UInt16)), @@gpu_attn_out, @@gpu_attn_ws,
          n, base_pos, @num_heads, heads_per_kv, @head_dim, @gpu_cache_cap, scale)
      else
        CUDA.attention_kv_f32(q, @gpu_k_cache.as(Pointer(Float32)),
          @gpu_v_cache.as(Pointer(Float32)), @@gpu_attn_out, @@gpu_attn_ws,
          n, base_pos, @num_heads, heads_per_kv, @head_dim, @gpu_cache_cap, scale)
      end
    end

    # Pure size policy for grow_dev_buf, split out so it can be asserted directly
    # without allocating anything or driving a GPU.
    #
    # Doubles to avoid realloc churn, but `cap_limit` clamps the doubling so a
    # budgeted buffer cannot overshoot its budget: without the clamp a buffer
    # sitting just under its limit doubles straight past it, making the effective
    # ceiling 2x the stated one. A single request larger than the limit still
    # wins, since under-allocating would corrupt the kernel's writes.
    def self.next_buf_cap(cur_cap : Int32, needed : Int32, cap_limit : Int32? = nil) : Int32
      doubled = cur_cap * 2
      doubled = cap_limit if cap_limit && doubled > cap_limit
      Math.max(needed, doubled)
    end

    # Grow-only device buffer; frees and reallocates only when too small.
    private def grow_dev_buf(ptr : Pointer(Float32), cur_cap : Int32, needed : Int32,
                             cap_limit : Int32? = nil) : {Pointer(Float32), Int32}
      return {ptr, cur_cap} if !ptr.null? && cur_cap >= needed
      CUDA.free(ptr.as(Pointer(Void))) unless ptr.null?
      new_cap = LlamaBlock.next_buf_cap(cur_cap, needed, cap_limit)
      np = Pointer(Float32).null
      CUDA.malloc(pointerof(np).as(Pointer(Pointer(Void))), new_cap.to_u64 * 4_u64)
      {np, new_cap}
    end

    # Diagnostics for the shared attention scratch. Chunked prefill must hold the
    # workspace to O(num_heads * chunk * total_len) rather than the unchunked
    # O(num_heads * seq^2); these let a spec assert that bound directly instead
    # of trusting it.
    def self.attn_ws_floats : Int32
      @@gpu_attn_ws_cap
    end

    def self.attn_staging_floats : Int32
      @@gpu_staging_cap
    end

    # --- GPU attention (full sequence, stays on device) ---
    private def attention_full_gpu(x : CudaMatrix) : CudaMatrix
      seq_len = x.rows
      head_dim = @head_dim
      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32

      # Big matmuls on GPU
      q_full = x * @w_q.as(CudaMatrix) # cuBLAS SGEMM
      k_full = x * @w_k.as(CudaMatrix)
      v_full = x * @w_v.as(CudaMatrix)

      # Bring to CPU for per-head RoPE + causal softmax (small matrices)
      q_full.sync_from_device!("attn_q") if q_full.device_dirty?
      k_full.sync_from_device!("attn_k") if k_full.device_dirty?
      v_full.sync_from_device!("attn_v") if v_full.device_dirty?
      add_bias!(q_full, @b_q)
      add_bias!(k_full, @b_k)
      add_bias!(v_full, @b_v)
      apply_head_rmsnorm!(q_full, @num_heads, @q_norm)
      apply_head_rmsnorm!(k_full, @num_kv_heads, @k_norm)

      output = SimpleMatrix.new(seq_len, @q_dim)
      heads_per_kv = @num_heads // @num_kv_heads

      @num_heads.times do |h|
        q_col = h * head_dim
        kv_h = h // heads_per_kv
        kv_col = kv_h * head_dim

        q_h = SimpleMatrix.new(seq_len, head_dim)
        k_h = SimpleMatrix.new(seq_len, head_dim)
        v_h = SimpleMatrix.new(seq_len, head_dim)
        seq_len.times do |s|
          head_dim.times do |d|
            q_h[s, d] = q_full[s, q_col + d]
            k_h[s, d] = k_full[s, kv_col + d]
            v_h[s, d] = v_full[s, kv_col + d]
          end
        end

        apply_rope!(q_h, 0)
        apply_rope!(k_h, 0)
        causal_attention!(output, q_h, k_h, v_h, q_col, scale)
      end

      # Output projection on GPU: convert to CudaMatrix, then SGEMM
      result = CudaMatrix.new(seq_len, @q_dim)
      seq_len.times { |i| @q_dim.times { |j| result[i, j] = output[i, j] } }
      result.sync_to_device!("attn_concat")
      result * @w_o.as(CudaMatrix)
    end

    # --- Helper: add a per-column bias vector to every row, in-place (host) ---
    private def add_bias!(m : SimpleMatrix, b : Array(Float32)?)
      return unless b
      raise ArgumentError.new("bias size #{b.size} does not match matrix cols #{m.cols}") unless b.size == m.cols
      bp = b.to_unsafe
      data = m.data.to_unsafe
      cols = m.cols
      r = 0
      while r < m.rows
        base = r * cols
        c = 0
        while c < cols
          data[base + c] += bp[c]
          c += 1
        end
        r += 1
      end
    end

    # CudaMatrix variant: used by the full-sequence GPU path after the
    # projection result has been synced back to host for per-head processing.
    private def add_bias!(m : CudaMatrix, b : Array(Float32)?)
      return unless b
      raise ArgumentError.new("bias size #{b.size} does not match matrix cols #{m.cols}") unless b.size == m.cols
      m.rows.times { |r| m.cols.times { |c| m[r, c] = (m[r, c].to_f32 + b[c]) } }
    end

    # --- Helper: Qwen3 QK-norm. Per-head RMSNorm over head_dim with a learned
    # weight, applied in-place to every (row, head) slice of `m`
    # [rows, nheads*head_dim], before RoPE. No-op when `weight` is nil (the
    # LLaMA/Qwen2 default), so existing models are byte-identical.
    private def apply_head_rmsnorm!(m : SimpleMatrix, nheads : Int32, weight : Array(Float32)?)
      return unless w = weight
      hd = @head_dim
      raise ArgumentError.new("qk_norm weight size #{w.size} != head_dim #{hd}") unless w.size == hd
      eps = @qk_norm_eps
      wp = w.to_unsafe
      data = m.data.to_unsafe
      cols = m.cols
      m.rows.times do |r|
        nheads.times do |h|
          base = r * cols + h * hd
          ss = 0.0_f32
          d = 0
          while d < hd
            v = data[base + d]
            ss += v * v
            d += 1
          end
          inv = (1.0 / Math.sqrt(ss / hd + eps)).to_f32
          d = 0
          while d < hd
            data[base + d] = data[base + d] * inv * wp[d]
            d += 1
          end
        end
      end
    end

    # CudaMatrix variant (full-sequence GPU path, after the projection has been
    # synced to host for per-head processing). Operates on the host mirror.
    private def apply_head_rmsnorm!(m : CudaMatrix, nheads : Int32, weight : Array(Float32)?)
      return unless w = weight
      hd = @head_dim
      raise ArgumentError.new("qk_norm weight size #{w.size} != head_dim #{hd}") unless w.size == hd
      eps = @qk_norm_eps
      m.rows.times do |r|
        nheads.times do |h|
          col0 = h * hd
          ss = 0.0_f32
          hd.times { |d| v = m[r, col0 + d].to_f32; ss += v * v }
          inv = (1.0 / Math.sqrt(ss / hd + eps)).to_f32
          hd.times { |d| m[r, col0 + d] = (m[r, col0 + d].to_f32 * inv * w[d]) }
        end
      end
    end

    # --- Helper: extract head slice ---
    private def extract_head(full : SimpleMatrix, rows : Int32, col_start : Int32, cols : Int32) : SimpleMatrix
      m = SimpleMatrix.new(rows, cols)
      rows.times { |r| cols.times { |c| m[r, c] = full[r, col_start + c] } }
      m
    end

    # --- Helper: inverse frequency for rotation index i (0..head_dim/2) ---
    private def inv_freq(i : Int32) : Float32
      if freqs = @rope_freqs
        freqs[i]
      else
        (1.0 / (@rope_theta ** (2.0 * i / @head_dim))).to_f32
      end
    end

    # --- Helper: apply RoPE in-place (HF half-split convention) ---
    private def apply_rope!(m : SimpleMatrix, start_pos : Int32)
      half = @head_dim // 2
      m.rows.times do |pos|
        actual_pos = pos + start_pos
        half.times do |i|
          freq = inv_freq(i)
          angle = (actual_pos * freq).to_f32
          cos_val = Math.cos(angle).to_f32
          sin_val = Math.sin(angle).to_f32
          x0 = m[pos, i].to_f32
          x1 = m[pos, i + half].to_f32
          m[pos, i] = x0 * cos_val - x1 * sin_val
          m[pos, i + half] = x1 * cos_val + x0 * sin_val
        end
      end
    end

    # --- Helper: causal attention, writes into output at q_col ---
    private def causal_attention!(output : SimpleMatrix, q_h : SimpleMatrix, k_h : SimpleMatrix, v_h : SimpleMatrix, q_col : Int32, scale : Float32)
      seq_len = q_h.rows
      k_t = k_h.transpose
      scores = q_h * k_t

      attn_weights = SimpleMatrix.new(seq_len, k_h.rows)
      seq_len.times do |i|
        max_val = -Float32::INFINITY
        (0..i).each { |j| sv = scores[i, j].to_f32 * scale; max_val = sv if sv > max_val }
        exp_sum = 0.0_f32
        (0..i).each do |j|
          e = Math.exp((scores[i, j].to_f32 * scale - max_val).to_f64).to_f32
          attn_weights[i, j] = e
          exp_sum += e
        end
        (0..i).each { |j| attn_weights[i, j] = attn_weights[i, j].to_f32 / exp_sum }
      end

      attn_out = attn_weights * v_h
      seq_len.times { |s| @head_dim.times { |d| output[s, q_col + d] = attn_out[s, d] } }
    end

    # --- Helper: cache flat array to matrix ---
    private def cache_to_matrix(cache : Array(Float32), rows : Int32, cols : Int32) : SimpleMatrix
      m = SimpleMatrix.new(rows, cols)
      rows.times { |r| cols.times { |c| m[r, c] = cache[r * cols + c] } }
      m
    end

    # --- Helper: matmul using GPU SGEMM if weights are CudaMatrix ---
    private def gpu_matmul(x : SimpleMatrix, w : SimpleMatrix | CudaMatrix | QuantizedWeight) : SimpleMatrix
      if w.is_a?(QuantizedWeight)
        if x.rows == 1
          # Decode (M=1): reuse persistent device buffers, no per-call alloc/free.
          xb = (@q8_in_bufs[x.cols] ||= CudaMatrix.new(1, x.cols))
          Profile.measure("gemm.in_h2d") do
            xb.raw_data.to_unsafe.copy_from(x.data.to_unsafe, x.cols)
            xb.mark_host_modified!
            xb.sync_to_device!("q8_gemm_in")
          end
          ob = (@q8_out_bufs[w.cols] ||= CudaMatrix.new(1, w.cols))
          Profile.measure("gemm.kernel") { w.gemv_into(xb, ob) }
          Profile.measure("gemm.out_d2h") { ob.sync_from_device!("q8_gemm_out") if ob.device_dirty? }
          result = SimpleMatrix.new(1, w.cols)
          Profile.measure("gemm.result_copy") do
            result.data.to_unsafe.copy_from(ob.raw_data.to_unsafe, w.cols)
          end
          result
        else
          # Prefill / batch (M>1): one-off allocation.
          x_gpu = CudaMatrix.new(x.rows, x.cols)
          x_gpu.raw_data.to_unsafe.copy_from(x.data.to_unsafe, x.rows * x.cols)
          x_gpu.sync_to_device!("q8_gemm_in")
          result_gpu = w.gemv(x_gpu)
          result_gpu.sync_from_device!("q8_gemm_out") if result_gpu.device_dirty?
          result = SimpleMatrix.new(result_gpu.rows, result_gpu.cols)
          result.data.to_unsafe.copy_from(result_gpu.raw_data.to_unsafe, result_gpu.rows * result_gpu.cols)
          x_gpu.free!
          result_gpu.free!
          result
        end
      elsif w.is_a?(CudaMatrix)
        # Convert input to GPU, GEMM, bring back
        x_gpu = CudaMatrix.new(x.rows, x.cols)
        x.rows.times { |r| x.cols.times { |c| x_gpu[r, c] = x[r, c] } }
        x_gpu.sync_to_device!("gemm_in")
        result_gpu = x_gpu * w # cuBLAS SGEMM
        result_gpu.sync_from_device!("gemm_out") if result_gpu.device_dirty?
        result = SimpleMatrix.new(result_gpu.rows, result_gpu.cols)
        result_gpu.rows.times { |r| result_gpu.cols.times { |c| result[r, c] = result_gpu[r, c].to_f32 } }
        x_gpu.free!
        result_gpu.free!
        result
      else
        x * w
      end
    end

    # Projection whose INPUT is already device-resident. Saves the upload that
    # gpu_matmul pays: on the device block chain the normed row is produced by the
    # RMSNorm kernel, so q/k/v need no host-to-device copy at all. The result still
    # comes back because RoPE, the KV host mirror and the attention staging are
    # host-side.
    private def gemv_from_device(x : CudaMatrix, w : QuantizedWeight) : SimpleMatrix
      ob = (@q8_out_bufs[w.cols] ||= CudaMatrix.new(1, w.cols))
      Profile.measure("gemm.kernel") { w.gemv_into(x, ob) }
      Profile.measure("gemm.out_d2h") { ob.sync_from_device!("dev_proj_out") if ob.device_dirty? }
      result = SimpleMatrix.new(1, w.cols)
      Profile.measure("gemm.result_copy") do
        result.data.to_unsafe.copy_from(ob.raw_data.to_unsafe, w.cols)
      end
      result
    end

    # Projection whose OUTPUT stays on the device. Used for o_proj, so the attention
    # result feeds the residual add without a readback.
    private def gemv_to_device(x : SimpleMatrix, w : QuantizedWeight, dst : CudaMatrix) : CudaMatrix
      xb = (@q8_in_bufs[x.cols] ||= CudaMatrix.new(1, x.cols))
      Profile.measure("gemm.in_h2d") do
        xb.raw_data.to_unsafe.copy_from(x.data.to_unsafe, x.cols)
        xb.mark_host_modified!
        xb.sync_to_device!("dev_proj_in")
      end
      Profile.measure("gemm.kernel") { w.gemv_into(xb, dst) }
      dst
    end
  end

  alias LlamaLayer = LlamaBlock
end
