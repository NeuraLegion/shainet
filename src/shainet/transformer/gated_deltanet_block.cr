module SHAInet
  # A Gated DeltaNet token-mixer block: the linear-attention layer type in a qwen3_5 hybrid
  # stack, standing in for LlamaBlock wherever layer_types says "linear_attention".
  #
  # Macro shape follows LlamaBlock exactly, so the two are interchangeable in a stack:
  #
  #   x -> norm1 -> mixer -> +x  ->  norm2 -> ffn -> +
  #
  # Only the mixer differs. Per the paper's block design (arXiv:2412.06464 section 3.4):
  #
  #   q, k: linear proj -> short conv -> SiLU -> L2 normalize
  #   v:    linear proj -> short conv -> SiLU
  #   a, b: linear proj only
  #   out:  gated delta rule -> norm -> SiLU gate -> output proj
  #
  # Their ablation says the short conv and the output gate both matter (removing either costs
  # more than removing the output norm), and that L2 normalization is essential -- so none of
  # those pieces are optional.
  #
  # WHAT THIS BUYS on a 16 GB card: no KV cache. The mixer's whole memory is a [d_v, d_k] state
  # per head plus a kernel-1 conv window, neither of which grows with context. Softmax attention
  # costs 96 KiB per position on the 30B, which is what caps context at 32768 here.
  #
  # UNVERIFIED against a checkpoint. The gate parameterization below is the documented Mamba2
  # form, but the paper says only "we use Mamba2's parameterization for alpha but omit it for
  # brevity", and there is no reference implementation on this machine to diff against. The
  # shapes, the state carry and the prefill/decode equivalence ARE specified and tested; the
  # exact gate formula needs a real Qwen3.5 checkpoint to confirm, which is why the loader still
  # refuses to build one of these from weights.
  class GatedDeltaNetBlock < MatrixLayer
    include QuantizedProjection

    getter d_model : Int32
    getter num_v_heads : Int32
    getter num_k_heads : Int32
    getter head_k : Int32
    getter head_v : Int32
    getter conv_kernel : Int32

    getter norm1 : RMSNorm
    getter norm2 : RMSNorm
    getter out_norm : RMSNorm
    getter ffn : SwiGLUFF

    # The five large projections carry the block's whole parameter cost and are quantized.
    #
    # w_alpha and w_beta are [d_model, num_v_heads] -- 0.5 MB against 100 MB for the rest -- so they
    # are NOT quantized, but they do move to the device as fp32. Computing them as a hand-rolled
    # triple loop cost 0.328 s per layer at a 1216-token prefill, 47% of the mixer once the
    # recurrence moved to the GPU, for what is simply two GEMMs. Precision matters here: alpha feeds
    # exp(-exp(a_log) * softplus(...)), so a Q4 rounding of the projection moves the decay itself.
    property w_q : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_k : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_v : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_o : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_gate : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_alpha : SimpleMatrix | CudaMatrix
    property w_beta : SimpleMatrix | CudaMatrix
    # Mamba2's per-head decay parameters. a_log is stored in log space because the decay must
    # stay in (0,1) for the state to be stable, and exp(-exp(a_log)*dt) is in (0,1) for any
    # real a_log and any dt > 0 -- no clamping needed.
    getter a_log : Array(Float64)
    getter dt_bias : Array(Float64)

    def initialize(@d_model : Int32, ff_hidden : Int32,
                   @num_v_heads : Int32 = 32, @num_k_heads : Int32 = 16,
                   @head_k : Int32 = 128, @head_v : Int32 = 128,
                   @conv_kernel : Int32 = 4, eps : Float64 = 1e-6, allocate : Bool = true)
      super(@d_model, SHAInet.none)
      raise ArgumentError.new("num_v_heads must be positive") unless @num_v_heads > 0
      raise ArgumentError.new("num_k_heads must be positive") unless @num_k_heads > 0
      # Value heads are grouped over key heads, as in grouped-query attention: several value
      # heads share one key head's state dimension.
      raise ArgumentError.new("num_v_heads (#{@num_v_heads}) must be divisible by num_k_heads (#{@num_k_heads})") unless @num_v_heads % @num_k_heads == 0

      @norm1 = RMSNorm.new(@d_model, eps)
      @norm2 = RMSNorm.new(@d_model, eps)
      # PER-HEAD output norm, over head_v rather than over the concatenated v_dim.
      #
      # Established from the real checkpoint, not from the paper: Qwen3.5-9B's
      # linear_attn.norm.weight is [128], which is head_v, against a v_dim of 4096. One weight
      # vector is shared across all 32 value heads and applied to each head's slice
      # independently. Normalizing the whole concatenation instead would couple the heads
      # through one shared RMS and silently change every output -- a wrong-but-plausible design
      # that no shape check would have caught.
      @out_norm = RMSNorm.new(@head_v, eps)
      @ffn = SwiGLUFF.new(@d_model, ff_hidden, allocate: allocate)

      k_dim = @num_k_heads * @head_k
      v_dim = @num_v_heads * @head_v
      if allocate
        @w_q = SimpleMatrix.new(@d_model, k_dim)
        @w_k = SimpleMatrix.new(@d_model, k_dim)
        @w_v = SimpleMatrix.new(@d_model, v_dim)
        @w_o = SimpleMatrix.new(v_dim, @d_model)
        @w_gate = SimpleMatrix.new(@d_model, v_dim)
        @w_alpha = SimpleMatrix.new(@d_model, @num_v_heads)
        @w_beta = SimpleMatrix.new(@d_model, @num_v_heads)
      else
        @w_q = SimpleMatrix.new(0, 0)
        @w_k = SimpleMatrix.new(0, 0)
        @w_v = SimpleMatrix.new(0, 0)
        @w_o = SimpleMatrix.new(0, 0)
        @w_gate = SimpleMatrix.new(0, 0)
        @w_alpha = SimpleMatrix.new(0, 0)
        @w_beta = SimpleMatrix.new(0, 0)
      end
      @a_log = Array(Float64).new(@num_v_heads, 0.0)
      @dt_bias = Array(Float64).new(@num_v_heads, 0.0)

      @conv_q = ShortConv.new(k_dim, @conv_kernel)
      @conv_k = ShortConv.new(k_dim, @conv_kernel)
      @conv_v = ShortConv.new(v_dim, @conv_kernel)

      @state = Array(SimpleMatrix?).new(@num_v_heads, nil)
      @conv_state_q = nil
      @conv_state_k = nil
      @conv_state_v = nil
    end

    getter conv_q : ShortConv
    getter conv_k : ShortConv
    getter conv_v : ShortConv

    @state : Array(SimpleMatrix?)
    @conv_state_q : SimpleMatrix?
    @conv_state_k : SimpleMatrix?
    @conv_state_v : SimpleMatrix?

    # Drop all recurrent state, so the next forward starts a fresh sequence.
    #
    # The analogue of clearing a KV cache, and it must clear the conv windows TOO: leaving them
    # behind would let the first positions of a new sequence see the tail of the previous one,
    # which is a silent correctness bug rather than a visible failure.
    def clear_cache!
      @num_v_heads.times { |h| @state[h] = nil }
      @conv_state_q = nil
      @conv_state_k = nil
      @conv_state_v = nil
      # The device state must go too, or a new sequence would continue from the previous one's
      # accumulated state -- silently wrong output rather than a visible failure, and exactly the
      # bug the conv-window clearing above exists to prevent.
      @dev_state.try(&.free!)
      @dev_state = nil
    end

    # Bytes of recurrent state held, for comparison against a KV cache.
    def state_bytes : Int64
      per_head = (@head_v * @head_k * 4).to_i64
      conv = ((@num_k_heads * @head_k * 2 + @num_v_heads * @head_v) * (@conv_kernel - 1) * 4).to_i64
      per_head * @num_v_heads + conv
    end

    # SiLU, x * sigmoid(x).
    private def silu(x : Float64) : Float64
      x / (1.0 + Math.exp(-x))
    end

    private def softplus(x : Float64) : Float64
      # log1p(exp(x)) guarded for large x, where exp overflows and the function is ~x.
      x > 20.0 ? x : Math.log1p(Math.exp(x))
    end

    # Per-head alpha and beta for each position.
    #
    # Mamba2's parameterization, which the paper adopts without restating:
    #
    #   dt    = softplus(x W_alpha + dt_bias)
    #   alpha = exp(-exp(a_log) * dt)      in (0,1) for any real a_log, dt > 0
    #   beta  = sigmoid(x W_beta)          in (0,1)
    #
    # Both ranges are structural rather than clamped, which matters: an alpha at or above 1
    # would let the state grow without bound over a long context, and a beta outside (0,1) would
    # break the delta rule's interpretation as a blend of old and new.
    def gates(normed : SimpleMatrix) : {Array(Array(Float64)), Array(Array(Float64))}
      seq = normed.rows
      # Two GEMMs, then elementwise. Previously a triple loop over seq x heads x d_model, which at a
      # 1216-token prefill was 0.328 s per layer -- 47% of the mixer once the recurrence moved to the
      # device -- for 159M iterations that cuBLAS does in milliseconds.
      a_proj = gpu_matmul(normed, @w_alpha)
      b_proj = gpu_matmul(normed, @w_beta)

      alpha = Array(Array(Float64)).new(@num_v_heads) { Array(Float64).new(seq, 0.0) }
      beta = Array(Array(Float64)).new(@num_v_heads) { Array(Float64).new(seq, 0.0) }
      @num_v_heads.times do |h|
        decay = Math.exp(@a_log[h])
        bias = @dt_bias[h]
        ah = alpha[h]
        bh = beta[h]
        seq.times do |t|
          ah[t] = Math.exp(-decay * softplus(a_proj[t, h].to_f64 + bias))
          bh[t] = 1.0 / (1.0 + Math.exp(-b_proj[t, h].to_f64))
        end
      end

      {alpha, beta}
    end

    # The mixer: normed input in, [seq, d_model] out.
    #
    # `chunk` selects the operator form. Above 1 it uses the chunked parallel path for prefill;
    # at 1 it is the sequential recurrence, which is all decode can use anyway. Both produce the
    # same numbers, which is specified rather than assumed.
    def mix(normed : SimpleMatrix, chunk : Int32 = 64) : SimpleMatrix
      seq = normed.rows
      k_dim = @num_k_heads * @head_k
      v_dim = @num_v_heads * @head_v

      q_lin = project(normed, @w_q, k_dim)
      k_lin = project(normed, @w_k, k_dim)
      v_lin = project(normed, @w_v, v_dim)

      q_c, @conv_state_q = @conv_q.forward(q_lin, @conv_state_q)
      k_c, @conv_state_k = @conv_k.forward(k_lin, @conv_state_k)
      v_c, @conv_state_v = @conv_v.forward(v_lin, @conv_state_v)

      seq.times do |t|
        k_dim.times { |j| q_c[t, j] = silu(q_c[t, j].to_f64); k_c[t, j] = silu(k_c[t, j].to_f64) }
        v_dim.times { |j| v_c[t, j] = silu(v_c[t, j].to_f64) }
      end

      alpha, beta = gates(normed)
      heads_per_k = @num_v_heads // @num_k_heads

      if dev = device_mix(q_c, k_c, v_c, alpha, beta, seq, k_dim, v_dim, heads_per_k)
        mixed = dev
      else
        mixed = SimpleMatrix.new(seq, v_dim, 0.0)
        @num_v_heads.times do |h|
          kh = h // heads_per_k
          qh = SimpleMatrix.new(seq, @head_k, 0.0)
          khm = SimpleMatrix.new(seq, @head_k, 0.0)
          vh = SimpleMatrix.new(seq, @head_v, 0.0)
          seq.times do |t|
            @head_k.times do |j|
              qh[t, j] = q_c[t, kh * @head_k + j]
              khm[t, j] = k_c[t, kh * @head_k + j]
            end
            @head_v.times { |j| vh[t, j] = v_c[t, h * @head_v + j] }
          end
          # L2 on q and k only, per the block design. The paper's ablation calls it essential, and
          # the recurrence's stability depends on ||k|| = 1.
          GatedDeltaNet.l2_normalize!(qh)
          GatedDeltaNet.l2_normalize!(khm)
          # Then scale q by 1/sqrt(head_k), as the reference kernel does unconditionally:
          #
          #   query = query / (query.shape[-1] ** 0.5)
          #
          # This is NOT cosmetic and it is NOT removed by the output norm, which is what I first
          # assumed. RMSNorm is scale-invariant only while the variance dominates its epsilon, and
          # here it does not: with the scale applied the core output's variance is ~1.9e-6 against an
          # eps of 1e-6, so eps is about half of it. Dropping the scale makes the state 128x larger in
          # variance, moves it out of the epsilon-dominated regime the trained weights expect, and
          # changes every value the norm produces.
          #
          # Measured against a numpy transcription of the reference on the real checkpoint, layer 0:
          # core rms 0.015739 without the scale against 0.001391 with it, a ratio of 11.31 = sqrt(128),
          # which is how the missing factor was identified.
          inv = 1.0 / Math.sqrt(@head_k.to_f64)
          seq.times { |t| @head_k.times { |j| qh[t, j] = qh[t, j].to_f64 * inv } }

          out_h, st = if chunk > 1
                        GatedDeltaNet.chunked(qh, khm, vh, alpha[h], beta[h], @state[h], chunk: chunk)
                      else
                        GatedDeltaNet.recurrent(qh, khm, vh, alpha[h], beta[h], @state[h])
                      end
          @state[h] = st
          seq.times { |t| @head_v.times { |j| mixed[t, h * @head_v + j] = out_h[t, j] } }
        end
      end

      # Output norm PER HEAD, then the SiLU gate, then project back to d_model.
      #
      # Each head's head_v slice is normalized on its own with the shared weight, matching the
      # checkpoint's [head_v] norm tensor. Doing it over the whole v_dim would make one head's
      # magnitude affect every other head's output.
      normed_mix = SimpleMatrix.new(seq, v_dim, 0.0)
      @num_v_heads.times do |h|
        slice = SimpleMatrix.new(seq, @head_v, 0.0)
        seq.times { |t| @head_v.times { |j| slice[t, j] = mixed[t, h * @head_v + j] } }
        normed_slice = @out_norm.forward(slice)
        seq.times { |t| @head_v.times { |j| normed_mix[t, h * @head_v + j] = normed_slice[t, j] } }
      end

      gate = project(normed, @w_gate, v_dim)
      seq.times do |t|
        v_dim.times { |j| normed_mix[t, j] = normed_mix[t, j].to_f64 * silu(gate[t, j].to_f64) }
      end
      project(normed_mix, @w_o, @d_model)
    end

    # Move the block's projections to the device, optionally quantized.
    #
    # The mixer's recurrence stays on the host deliberately. Per token it is ~2M FLOP (32 heads
    # over a 128x128 state) against ~600M for the projections, so a device kernel for it would
    # chase 0.3% of the work while the GEMMs are what make the block unusable in fp32. Quantizing
    # the projections is also what lets a hybrid stack load at all: they are 100 MB per layer
    # against 0.5 MB for everything else here.
    def to_gpu!(quantize : Bool = false, bits : Int32 = 8, offload : Bool = false)
      return unless CUDA.fully_available?
      if quantize
        @w_q = to_quant(@w_q, bits, offload)
        @w_k = to_quant(@w_k, bits, offload)
        @w_v = to_quant(@w_v, bits, offload)
        @w_o = to_quant(@w_o, bits, offload)
        @w_gate = to_quant(@w_gate, bits, offload)
      else
        raise ArgumentError.new("dense offload requires quantization (offload is Q4-only)") if offload
        @w_q = @w_q.as(SimpleMatrix).to_cuda if @w_q.is_a?(SimpleMatrix)
        @w_k = @w_k.as(SimpleMatrix).to_cuda if @w_k.is_a?(SimpleMatrix)
        @w_v = @w_v.as(SimpleMatrix).to_cuda if @w_v.is_a?(SimpleMatrix)
        @w_o = @w_o.as(SimpleMatrix).to_cuda if @w_o.is_a?(SimpleMatrix)
        @w_gate = @w_gate.as(SimpleMatrix).to_cuda if @w_gate.is_a?(SimpleMatrix)
      end
      # fp32 on the device, never quantized -- see the property comment.
      @w_alpha = @w_alpha.as(SimpleMatrix).to_cuda if @w_alpha.is_a?(SimpleMatrix)
      @w_beta = @w_beta.as(SimpleMatrix).to_cuda if @w_beta.is_a?(SimpleMatrix)
      @norm1.to_gpu!
      @norm2.to_gpu!
      @out_norm.to_gpu!
      @ffn.to_gpu!(quantize, bits, offload)
    end

    # True once the projections are quantized, so a caller can tell a loaded-and-quantized block
    # from one still holding fp32 without inspecting weight types itself.
    def quantized? : Bool
      @w_q.is_a?(QuantizedWeight) && @w_k.is_a?(QuantizedWeight) && @w_v.is_a?(QuantizedWeight) &&
        @w_o.is_a?(QuantizedWeight) && @w_gate.is_a?(QuantizedWeight)
    end

    # The recurrence on the device: one kernel launch for the whole sequence, all heads.
    #
    # Returns nil when the kernel is unavailable, so the host path above stays the reference
    # implementation and the CPU-only build keeps working. The two are pinned equivalent by spec.
    #
    # The kernel does the per-head slicing, the L2 normalization and the q scale itself, so nothing
    # here reshapes per head -- that copying cost as much as the arithmetic it fed. What remains is
    # one upload of q/k/v and the gates, one download of the output, and the state kept on the
    # device across calls so a decode step does not re-upload it.
    private def device_mix(q_c : SimpleMatrix, k_c : SimpleMatrix, v_c : SimpleMatrix,
                           alpha : Array(Array(Float64)), beta : Array(Array(Float64)),
                           seq : Int32, k_dim : Int32, v_dim : Int32,
                           heads_per_k : Int32) : SimpleMatrix?
      # SHAINET_GDN_DEVICE=0 forces the host path. Read per call rather than memoized so a spec can
      # A/B the two within one process, which is how their equivalence is asserted.
      return if ENV.fetch("SHAINET_GDN_DEVICE", "1") == "0"
      return unless CUDA.gated_delta_rule_available?

      qd = CudaMatrix.new(seq, k_dim)
      kd = CudaMatrix.new(seq, k_dim)
      vd = CudaMatrix.new(seq, v_dim)
      ad = CudaMatrix.new(seq, @num_v_heads)
      bd = CudaMatrix.new(seq, @num_v_heads)
      od = CudaMatrix.new(seq, v_dim)
      begin
        qd.raw_data.to_unsafe.copy_from(q_c.data.to_unsafe, seq * k_dim)
        kd.raw_data.to_unsafe.copy_from(k_c.data.to_unsafe, seq * k_dim)
        vd.raw_data.to_unsafe.copy_from(v_c.data.to_unsafe, seq * v_dim)
        seq.times do |t|
          @num_v_heads.times do |h|
            ad[t, h] = alpha[h][t]
            bd[t, h] = beta[h][t]
          end
        end
        qd.sync_to_device!("gdn_q")
        kd.sync_to_device!("gdn_k")
        vd.sync_to_device!("gdn_v")
        ad.sync_to_device!("gdn_alpha")
        bd.sync_to_device!("gdn_beta")

        st = device_state
        CUDA.gated_delta_rule(
          qd.device_ptr.not_nil!, kd.device_ptr.not_nil!, vd.device_ptr.not_nil!,
          ad.device_ptr.not_nil!, bd.device_ptr.not_nil!,
          st.device_ptr.not_nil!, od.device_ptr.not_nil!,
          seq, @num_v_heads, @num_k_heads, @head_k, @head_v, heads_per_k,
          (1.0 / Math.sqrt(@head_k.to_f64)).to_f32)
        od.mark_device_dirty!
        od.sync_from_device!("gdn_out")

        result = SimpleMatrix.new(seq, v_dim)
        result.data.to_unsafe.copy_from(od.raw_data.to_unsafe, seq * v_dim)
        result
      ensure
        qd.free!
        kd.free!
        vd.free!
        ad.free!
        bd.free!
        od.free!
      end
    end

    # The [num_v_heads, head_k, head_v] recurrent state, resident on the device.
    #
    # Allocated zeroed on first use, which is exactly the fresh-sequence state, and reused across
    # calls so a decode step carries the state without a round trip. clear_cache! frees it, so a new
    # sequence starts from zeros again rather than from the previous one's tail.
    private def device_state : CudaMatrix
      st = @dev_state
      return st if st
      st = CudaMatrix.new(@num_v_heads * @head_k, @head_v)
      st.zero!
      st.sync_to_device!("gdn_state_init")
      @dev_state = st
    end

    @dev_state : CudaMatrix?

    private def project(x : SimpleMatrix, w : SimpleMatrix | CudaMatrix | QuantizedWeight, out_dim : Int32) : SimpleMatrix
      dst = gpu_matmul(x, w)
      raise "projection produced #{dst.cols} columns, expected #{out_dim}" unless dst.cols == out_dim
      dst
    end

    def forward(x : SimpleMatrix) : SimpleMatrix
      h = x + mix(@norm1.forward(x))
      h + ffn_forward(@norm2.forward(h))
    end

    # Single-token step, for decode. Same math, chunk 1.
    def forward_cached(x : SimpleMatrix) : SimpleMatrix
      h = x + mix(@norm1.forward(x), chunk: 1)
      h + ffn_forward(@norm2.forward(h))
    end

    # Run the FFN via the device batch path when the weights are quantized, avoiding the
    # host-side SwiGLU loop (9.6 s of a 103 s prefill on the 27B). The device path streams
    # each Q4 weight once, applies SiLU on-device via the CUDA kernel, and reads back the
    # result -- identical numerics, ~10x faster for the activation.
    private def ffn_forward(normed : SimpleMatrix) : SimpleMatrix
      ffn = @ffn
      return ffn.forward(normed) unless ffn.gate_proj.is_a?(QuantizedWeight) && CUDA.fully_available?

      n = normed.rows
      d = normed.cols
      out_cols = ffn.down_proj.as(QuantizedWeight).cols

      # Upload normed to device.
      xd = CudaMatrix.new(n, d)
      xd.raw_data.to_unsafe.copy_from(normed.data.to_unsafe, n * d)
      xd.mark_host_modified!
      xd.sync_to_device!("gdn_ffn_in")

      od = CudaMatrix.new(n, out_cols)
      ffn.forward_device_batch(xd, od)
      od.sync_from_device!("gdn_ffn_out")

      result = SimpleMatrix.new(n, out_cols)
      result.data.to_unsafe.copy_from(od.raw_data.to_unsafe, n * out_cols)
      xd.free!
      od.free!
      result
    end

    def backward(d_out : SimpleMatrix) : SimpleMatrix
      raise NotImplementedError.new("GatedDeltaNetBlock is inference-only for now")
    end

    def apply_gradients(lr : Float64)
      raise NotImplementedError.new("GatedDeltaNetBlock is inference-only for now")
    end
  end
end
