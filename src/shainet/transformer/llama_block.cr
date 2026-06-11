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

    # KV cache: stored per kv_head as [seq_len, head_dim] growing matrices
    @k_cache : Array(Array(Float32)) # [num_kv_heads][seq_len * head_dim]
    @v_cache : Array(Array(Float32))
    @cache_len : Int32 = 0

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

    def clear_cache!
      @k_cache.each(&.clear)
      @v_cache.each(&.clear)
      @cache_len = 0
    end

    def to_gpu!(quantize : Bool = false)
      return unless CUDA.fully_available?
      if quantize
        @w_q = QuantizedCudaMatrix.from_simple(@w_q.as(SimpleMatrix)) if @w_q.is_a?(SimpleMatrix)
        @w_k = QuantizedCudaMatrix.from_simple(@w_k.as(SimpleMatrix)) if @w_k.is_a?(SimpleMatrix)
        @w_v = QuantizedCudaMatrix.from_simple(@w_v.as(SimpleMatrix)) if @w_v.is_a?(SimpleMatrix)
        @w_o = QuantizedCudaMatrix.from_simple(@w_o.as(SimpleMatrix)) if @w_o.is_a?(SimpleMatrix)
      else
        @w_q = @w_q.as(SimpleMatrix).to_cuda unless @w_q.is_a?(CudaMatrix)
        @w_k = @w_k.as(SimpleMatrix).to_cuda unless @w_k.is_a?(CudaMatrix)
        @w_v = @w_v.as(SimpleMatrix).to_cuda unless @w_v.is_a?(CudaMatrix)
        @w_o = @w_o.as(SimpleMatrix).to_cuda unless @w_o.is_a?(CudaMatrix)
      end
      @norm1.to_gpu!
      @norm2.to_gpu!
      @ffn.to_gpu!(quantize)
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
      heads_per_kv = @num_heads // @num_kv_heads

      @num_heads.times do |h|
        q_col = h * head_dim
        kv_h = h // heads_per_kv

        # Q for new tokens only
        q_h = extract_head(q_full, new_tokens, q_col, head_dim)
        apply_rope!(q_h, start_pos)

        # K/V from cache (K already has RoPE applied at insert time)
        k_h = cache_to_matrix(@k_cache[kv_h], total_len, head_dim)
        v_h = cache_to_matrix(@v_cache[kv_h], total_len, head_dim)

        # Attention: new Q [new_tokens, hd] @ cached K^T [hd, total_len]
        k_t = k_h.transpose
        scores = q_h * k_t # [new_tokens, total_len]

        # Causal softmax (each new token can see all cached + itself)
        attn_weights = SimpleMatrix.new(new_tokens, total_len)
        new_tokens.times do |i|
          visible = start_pos + i + 1 # this token can see positions 0..start_pos+i
          max_val = -Float32::INFINITY
          visible.times { |j| sv = scores[i, j].to_f32 * scale; max_val = sv if sv > max_val }
          exp_sum = 0.0_f32
          visible.times do |j|
            e = Math.exp((scores[i, j].to_f32 * scale - max_val).to_f64).to_f32
            attn_weights[i, j] = e
            exp_sum += e
          end
          visible.times { |j| attn_weights[i, j] = attn_weights[i, j].to_f32 / exp_sum }
        end

        attn_out = attn_weights * v_h # [new_tokens, head_dim]
        new_tokens.times do |s|
          head_dim.times { |d| output[s, q_col + d] = attn_out[s, d] }
        end
      end

      gpu_matmul(output, @w_o)
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
        # Quantized GPU GEMV: upload activations, dequant-in-kernel matmul, bring back
        x_gpu = CudaMatrix.new(x.rows, x.cols)
        x.rows.times { |r| x.cols.times { |c| x_gpu[r, c] = x[r, c] } }
        x_gpu.sync_to_device!("q8_gemm_in")
        result_gpu = w.gemv(x_gpu)
        result_gpu.sync_from_device!("q8_gemm_out") if result_gpu.device_dirty?
        result = SimpleMatrix.new(result_gpu.rows, result_gpu.cols)
        result_gpu.rows.times { |r| result_gpu.cols.times { |c| result[r, c] = result_gpu[r, c].to_f32 } }
        result
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
