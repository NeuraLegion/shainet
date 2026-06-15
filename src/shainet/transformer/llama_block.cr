require "../basic/matrix_layer"

module SHAInet
  # LLaMA-style transformer block with KV cache for efficient generation.
  # Supports Grouped Query Attention (GQA).
  class LlamaBlock < MatrixLayer
    getter norm1 : RMSNorm
    getter norm2 : RMSNorm
    getter ffn : SwiGLUFF
    getter num_heads : Int32
    getter num_kv_heads : Int32
    getter head_dim : Int32
    getter d_model : Int32
    property rope_theta : Float64
    # Optional precomputed inverse frequencies (size head_dim/2). When set,
    # these override the default theta^(-2i/d) computation (used for LLaMA 3
    # rope_scaling). nil means use the default.
    property rope_freqs : Array(Float32)? = nil

    property w_q : SimpleMatrix | CudaMatrix | QuantizedCudaMatrix
    property w_k : SimpleMatrix | CudaMatrix | QuantizedCudaMatrix
    property w_v : SimpleMatrix | CudaMatrix | QuantizedCudaMatrix
    property w_o : SimpleMatrix | CudaMatrix | QuantizedCudaMatrix

    # Optional Q/K/V projection biases. Qwen2-style architectures add a bias to
    # the query/key/value projections; LLaMA/Mistral do not. Kept as host-side
    # fp32 vectors and added to the projection output before RoPE, so they work
    # identically on the fp32, CUDA, and Q8 weight paths. nil means "no bias"
    # (the LLaMA default — zero overhead, behaviour unchanged). Sizes: b_q is
    # d_model; b_k/b_v are num_kv_heads * head_dim. o_proj has no bias in Qwen2.
    property b_q : Array(Float32)? = nil
    property b_k : Array(Float32)? = nil
    property b_v : Array(Float32)? = nil

    # KV cache: stored per kv_head as [seq_len, head_dim] growing matrices
    @k_cache : Array(Array(Float32)) # [num_kv_heads][seq_len * head_dim]
    @v_cache : Array(Array(Float32))
    @cache_len : Int32 = 0

    # GPU-resident mirror of the KV cache, laid out [num_kv_heads, capacity,
    # head_dim] per tensor. The CPU cache stays the source of truth: the GPU
    # copy is appended incrementally each token and fully re-uploaded from the
    # mirror whenever capacity grows. Only used when CUDA kernels are loaded.
    @gpu_k_cache : Pointer(Float32) = Pointer(Float32).null
    @gpu_v_cache : Pointer(Float32) = Pointer(Float32).null
    @gpu_cache_cap : Int32 = 0
    # Persistent device buffers for the attention hot path (grow-only, never
    # freed mid-inference so they cannot be GC-collected during a kernel).
    @gpu_staging : Pointer(Float32) = Pointer(Float32).null
    @gpu_staging_cap : Int32 = 0
    @gpu_attn_out : Pointer(Float32) = Pointer(Float32).null
    @gpu_attn_out_cap : Int32 = 0
    @gpu_attn_ws : Pointer(Float32) = Pointer(Float32).null
    @gpu_attn_ws_cap : Int32 = 0
    @staging_host : Array(Float32) = Array(Float32).new
    @gpu_attn_avail : Bool? = nil
    # Force the CPU attention path even when CUDA is available (tests/fallback).
    property force_cpu_attention : Bool = false

    def initialize(@d_model : Int32, @num_heads : Int32, ff_hidden : Int32,
                   eps : Float64 = 1e-6, @rope_theta : Float64 = 10000.0,
                   @num_kv_heads : Int32 = @num_heads)
      super(@d_model, SHAInet.none)
      raise ArgumentError.new("d_model must be divisible by num_heads") unless @d_model % @num_heads == 0
      raise ArgumentError.new("num_heads must be divisible by num_kv_heads") unless @num_kv_heads > 0 && @num_heads % @num_kv_heads == 0
      @head_dim = @d_model // @num_heads
      kv_dim = @num_kv_heads * @head_dim
      @norm1 = RMSNorm.new(@d_model, eps)
      @norm2 = RMSNorm.new(@d_model, eps)
      @ffn = SwiGLUFF.new(@d_model, ff_hidden)
      @w_q = SimpleMatrix.new(@d_model, @d_model)
      @w_k = SimpleMatrix.new(@d_model, kv_dim)
      @w_v = SimpleMatrix.new(@d_model, kv_dim)
      @w_o = SimpleMatrix.new(@d_model, @d_model)
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
      {@gpu_k_cache, @gpu_v_cache, @gpu_staging, @gpu_attn_out, @gpu_attn_ws}.each do |p|
        CUDA.free(p.as(Pointer(Void))) unless p.null?
      end
    end

    def to_gpu!(quantize : Bool = false)
      return unless CUDA.fully_available?
      if quantize
        @w_q = to_q8(@w_q)
        @w_k = to_q8(@w_k)
        @w_v = to_q8(@w_v)
        @w_o = to_q8(@w_o)
      else
        # Only promote host weights; leave existing CudaMatrix/QuantizedCudaMatrix as-is.
        @w_q = @w_q.as(SimpleMatrix).to_cuda if @w_q.is_a?(SimpleMatrix)
        @w_k = @w_k.as(SimpleMatrix).to_cuda if @w_k.is_a?(SimpleMatrix)
        @w_v = @w_v.as(SimpleMatrix).to_cuda if @w_v.is_a?(SimpleMatrix)
        @w_o = @w_o.as(SimpleMatrix).to_cuda if @w_o.is_a?(SimpleMatrix)
      end
      @norm1.to_gpu!
      @norm2.to_gpu!
      @ffn.to_gpu!(quantize)
    end

    # Quantize a weight to Q8 regardless of its current representation,
    # avoiding mixed precision/quantization state on repeated to_gpu! calls.
    private def to_q8(w : SimpleMatrix | CudaMatrix | QuantizedCudaMatrix) : QuantizedCudaMatrix
      case w
      when QuantizedCudaMatrix then w
      when CudaMatrix          then QuantizedCudaMatrix.from_simple(w.to_simple)
      else                          QuantizedCudaMatrix.from_simple(w.as(SimpleMatrix))
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
      normed = @norm1.forward(x)
      attn = attention_cached_cpu(normed)
      h = x + attn
      normed2 = @norm2.forward(h)
      ff_out = @ffn.forward(normed2)
      h + ff_out
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

      output = SimpleMatrix.new(seq_len, @d_model)
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

      # Apply RoPE to new K at insert time (HF half-split), then append to cache.
      half = head_dim // 2
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

      total_len = @cache_len + new_tokens
      @cache_len = total_len
      output = SimpleMatrix.new(new_tokens, @d_model)

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
      dm = @d_model
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
      dm = @d_model
      half = head_dim // 2
      heads_per_kv = @num_heads // @num_kv_heads
      chunk = new_tokens * head_dim     # floats per kv_head per tensor
      kv_floats = @num_kv_heads * chunk # staging size of K (and of V)
      q_floats = new_tokens * dm
      staging_floats = 2 * kv_floats + q_floats
      ws_floats = @num_heads * new_tokens * total_len

      ensure_gpu_cache!(total_len)
      @gpu_staging, @gpu_staging_cap = grow_dev_buf(@gpu_staging, @gpu_staging_cap, staging_floats)
      @gpu_attn_out, @gpu_attn_out_cap = grow_dev_buf(@gpu_attn_out, @gpu_attn_out_cap, q_floats)
      @gpu_attn_ws, @gpu_attn_ws_cap = grow_dev_buf(@gpu_attn_ws, @gpu_attn_ws_cap, ws_floats)

      st = @staging_host
      if st.size < staging_floats
        st = Array(Float32).new(staging_floats, 0.0_f32)
        @staging_host = st
      end
      stp = st.to_unsafe

      # New K/V rows: each kv_head's tail is contiguous in the CPU mirror
      # (RoPE already applied to K at insert).
      tail = start_pos * head_dim
      @num_kv_heads.times do |kv_h|
        (stp + kv_h * chunk).copy_from(@k_cache[kv_h].to_unsafe + tail, chunk)
        (stp + kv_floats + kv_h * chunk).copy_from(@v_cache[kv_h].to_unsafe + tail, chunk)
      end

      # RoPE-rotate Q (HF half-split) into the staging blob, token-major.
      # cos/sin depend only on (pos, rotation index), so compute once per token.
      qptr = q_full.data.to_unsafe
      qst = stp + 2 * kv_floats
      cosv = Array(Float32).new(half, 0.0_f32)
      sinv = Array(Float32).new(half, 0.0_f32)
      cp = cosv.to_unsafe
      sp = sinv.to_unsafe
      new_tokens.times do |i|
        pos = start_pos + i
        half.times do |r|
          angle = (pos * inv_freq(r)).to_f32
          cp[r] = Math.cos(angle).to_f32
          sp[r] = Math.sin(angle).to_f32
        end
        row = i * dm
        @num_heads.times do |h|
          base = row + h * head_dim
          r = 0
          while r < half
            x0 = qptr[base + r]
            x1 = qptr[base + r + half]
            qst[base + r] = x0 * cp[r] - x1 * sp[r]
            qst[base + r + half] = x1 * cp[r] + x0 * sp[r]
            r += 1
          end
        end
      end

      CUDA.memcpy(@gpu_staging.as(Pointer(Void)), stp.as(Pointer(Void)),
        staging_floats.to_u64 * 4_u64, CUDA::MemcpyKind::HostToDevice)
      CUDA.kv_cache_append_f32(@gpu_staging, @gpu_k_cache, @gpu_v_cache,
        new_tokens, start_pos, @num_kv_heads, head_dim, @gpu_cache_cap)
      CUDA.attention_kv_f32(@gpu_staging + 2 * kv_floats, @gpu_k_cache, @gpu_v_cache,
        @gpu_attn_out, @gpu_attn_ws, new_tokens, start_pos,
        @num_heads, heads_per_kv, head_dim, @gpu_cache_cap, scale)
      # Synchronous D2H read-back also orders after both kernels above.
      CUDA.memcpy(output.data.to_unsafe.as(Pointer(Void)), @gpu_attn_out.as(Pointer(Void)),
        q_floats.to_u64 * 4_u64, CUDA::MemcpyKind::DeviceToHost)
    end

    # GPU attention is used whenever the CUDA kernels are loadable and the
    # caller has not forced the CPU path (property or SHAINET_CPU_ATTENTION=1).
    # Memoized: fully_available? dlopens.
    private def gpu_attention? : Bool
      return false if @force_cpu_attention
      avail = @gpu_attn_avail
      if avail.nil?
        avail = !ENV["SHAINET_CPU_ATTENTION"]? && CUDA.fully_available?
        @gpu_attn_avail = avail
      end
      avail
    end

    # Ensure the device KV cache holds at least total_len positions per
    # kv_head. Grows by doubling; on growth the CPU mirror (which already
    # contains the new tokens) is re-uploaded into the fresh buffers.
    private def ensure_gpu_cache!(total_len : Int32)
      return if @gpu_cache_cap >= total_len
      head_dim = @head_dim
      new_cap = Math.max(256, Math.max(total_len, @gpu_cache_cap * 2))
      bytes = @num_kv_heads.to_u64 * new_cap.to_u64 * head_dim.to_u64 * 4_u64
      CUDA.free(@gpu_k_cache.as(Pointer(Void))) unless @gpu_k_cache.null?
      CUDA.free(@gpu_v_cache.as(Pointer(Void))) unless @gpu_v_cache.null?
      kp = Pointer(Float32).null
      vp = Pointer(Float32).null
      CUDA.malloc(pointerof(kp).as(Pointer(Pointer(Void))), bytes)
      CUDA.malloc(pointerof(vp).as(Pointer(Pointer(Void))), bytes)
      @gpu_k_cache = kp
      @gpu_v_cache = vp
      @gpu_cache_cap = new_cap
      @num_kv_heads.times do |kv_h|
        used = @k_cache[kv_h].size
        next if used == 0
        dst_off = kv_h.to_i64 * new_cap * head_dim
        CUDA.memcpy((kp + dst_off).as(Pointer(Void)), @k_cache[kv_h].to_unsafe.as(Pointer(Void)),
          used.to_u64 * 4_u64, CUDA::MemcpyKind::HostToDevice)
        CUDA.memcpy((vp + dst_off).as(Pointer(Void)), @v_cache[kv_h].to_unsafe.as(Pointer(Void)),
          used.to_u64 * 4_u64, CUDA::MemcpyKind::HostToDevice)
      end
    end

    # Grow-only device buffer; frees and reallocates only when too small.
    private def grow_dev_buf(ptr : Pointer(Float32), cur_cap : Int32, needed : Int32) : {Pointer(Float32), Int32}
      return {ptr, cur_cap} if !ptr.null? && cur_cap >= needed
      CUDA.free(ptr.as(Pointer(Void))) unless ptr.null?
      new_cap = Math.max(needed, cur_cap * 2)
      np = Pointer(Float32).null
      CUDA.malloc(pointerof(np).as(Pointer(Pointer(Void))), new_cap.to_u64 * 4_u64)
      {np, new_cap}
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

      output = SimpleMatrix.new(seq_len, @d_model)
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
      result = CudaMatrix.new(seq_len, @d_model)
      seq_len.times { |i| @d_model.times { |j| result[i, j] = output[i, j] } }
      result.sync_to_device!("attn_concat")
      result * @w_o.as(CudaMatrix)
    end

    # --- Helper: add a per-column bias vector to every row, in-place (host) ---
    private def add_bias!(m : SimpleMatrix, b : Array(Float32)?)
      return unless b
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
      m.rows.times { |r| m.cols.times { |c| m[r, c] = (m[r, c].to_f32 + b[c]) } }
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
    private def gpu_matmul(x : SimpleMatrix, w : SimpleMatrix | CudaMatrix | QuantizedCudaMatrix) : SimpleMatrix
      if w.is_a?(QuantizedCudaMatrix)
        if x.rows == 1
          # Decode (M=1): reuse persistent device buffers, no per-call alloc/free.
          xb = (@q8_in_bufs[x.cols] ||= CudaMatrix.new(1, x.cols))
          xb.raw_data.to_unsafe.copy_from(x.data.to_unsafe, x.cols)
          xb.mark_host_modified!
          xb.sync_to_device!("q8_gemm_in")
          ob = (@q8_out_bufs[w.cols] ||= CudaMatrix.new(1, w.cols))
          w.gemv_into(xb, ob)
          ob.sync_from_device!("q8_gemm_out") if ob.device_dirty?
          result = SimpleMatrix.new(1, w.cols)
          result.data.to_unsafe.copy_from(ob.raw_data.to_unsafe, w.cols)
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
        result
      else
        x * w
      end
    end
  end

  alias LlamaLayer = LlamaBlock
end
