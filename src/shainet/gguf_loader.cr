require "./gguf"

module SHAInet
  module HFLoader
    # Load a model from a GGUF file (llama.cpp / Ollama format).
    #
    # The GGUF contains everything: architecture metadata, tokenizer, and
    # pre-quantized weights (Q4_K / Q6_K / F32). No config.json or tokenizer.json
    # needed. Weights stay in their native k-quant format on the device -- no
    # transcoding to our Q4 format.
    #
    # Supported architectures: qwen35 (Qwen3.5 / Qwen3.8 hybrid stack).
    def self.load_gguf(path : String) : Network
      gf = GGUF::File.open(path)
      begin
        arch = gf.meta_string("general.architecture") || raise "GGUF missing general.architecture"
        raise "unsupported GGUF architecture: #{arch} (expected qwen35)" unless arch == "qwen35"

        d = (gf.meta_u32("#{arch}.embedding_length") || raise "missing embedding_length").to_i32
        n_layers = (gf.meta_u32("#{arch}.block_count") || raise "missing block_count").to_i32
        ff = (gf.meta_u32("#{arch}.feed_forward_length") || raise "missing feed_forward_length").to_i32
        n_heads = (gf.meta_u32("#{arch}.attention.head_count") || raise "missing head_count").to_i32
        n_kv_heads = (gf.meta_u32("#{arch}.attention.head_count_kv") || 4).to_i32
        head_dim = (gf.meta_u32("#{arch}.attention.key_length") || (d // n_heads)).to_i32
        eps = (gf.meta_f32("#{arch}.attention.layer_norm_rms_epsilon") || 1e-6_f32).to_f64
        rope_theta = (gf.meta_f32("#{arch}.rope.freq_base") || 10_000_000.0_f32).to_f64
        conv_kernel = (gf.meta_u32("#{arch}.ssm.conv_kernel") || 4).to_i32
        full_attn_interval = (gf.meta_u32("#{arch}.full_attention_interval") || 4).to_i32
        partial_rotary = (gf.meta_u32("#{arch}.rope.dimension_count") || 64).to_i32

        # The GGUF block_count includes MTP layers; the actual transformer layers are n_layers - nextn
        nextn = (gf.meta_u32("#{arch}.nextn_predict_layers") || 0).to_i32
        num_transformer_layers = n_layers - nextn

        # Build the layer type list: every full_attn_interval-th layer is full_attention
        types = Array(String).new(num_transformer_layers) do |i|
          (i > 0 && (i + 1) % full_attn_interval == 0) ? "full_attention" : "linear_attention"
        end

        # SSM (DeltaNet) parameters
        ssm_heads = (gf.meta_u32("#{arch}.ssm.time_step_rank") || 48).to_i32
        ssm_state = (gf.meta_u32("#{arch}.ssm.state_size") || 128).to_i32
        ssm_inner = (gf.meta_u32("#{arch}.ssm.inner_size") || 6144).to_i32
        ssm_k_heads = (gf.meta_u32("#{arch}.ssm.group_count") || 16).to_i32
        ssm_head_dim = ssm_inner // ssm_heads

        Log.info { "gguf: #{arch} #{num_transformer_layers} layers, d=#{d}, ff=#{ff}, heads=#{n_heads}/#{n_kv_heads}" }

        net = Network.new
        net.add_layer(:input, 1)
        net.add_layer(:embedding, d, vocab_size: 1) # Dummy; real embedding set below # Qwen3.5 vocab

        # Embedding: may be Q4_K or F32 in GGUF
        emb_layer = net.hidden_layers.find(&.is_a?(EmbeddingLayer)).as(EmbeddingLayer)
        emb_info = gf.tensors["token_embd.weight"]
        load_gguf_embedding(gf, emb_info, emb_layer, d)
        # If q4k_embedding was set, the 5 GB fp32 matrix from add_layer is never used.
        # Replace it with a tiny dummy to free the memory immediately.
        if emb_layer.q4k_embedding
          emb_layer.embeddings = SimpleMatrix.new(1, d)
          GC.collect
        end

        @@progress.try &.call(0, num_transformer_layers)

        # Layer-level GPU/CPU split: fill GPU back-to-front until the VRAM budget is met.
        # Layers that don't fit stay host-resident and run the CPU k-quant kernels.
        #
        # Sizes come from the GGUF rather than a per-layer average. The averages this replaced
        # (237 MB per layer, 535 MB for lm_head) were wrong in both directions on Qwen3.8-27B:
        # layers actually run 199.7-240.6 MB (mean 223.2 GDN / 211.3 full) and output.weight is
        # 994.6 MB, not 535. The lm_head error alone claimed 460 MB that does not exist, and the
        # two errors only partly cancel: measured 51 layers placed with 2505 MB still free, which
        # is ~11 more layers' worth of headroom left unused while those layers ran on the CPU at
        # 3.3x the per-layer cost.
        layer_bytes = Array(UInt64).new(num_transformer_layers, 0_u64)
        gf.tensors.each do |name, info|
          next unless name.starts_with?("blk.")
          idx = name.split('.')[1].to_i? || next
          next unless idx < num_transformer_layers
          layer_bytes[idx] += info.byte_size
        end
        lm_head_bytes = gf.tensors["output.weight"].byte_size

        gpu_layers = num_transformer_layers
        # The reserve is not slack: it has to cover everything that is NOT weights, and the parts
        # that matter scale with CONTEXT. Sizing it as a constant is what made a long conversation
        # die mid-turn -- 52 layers placed, then an OOM around 8000 tokens on a card that had loaded
        # the model fine. Derive it from the context we intend to support instead, so the trade is
        # made once at load (fewer layers, slower per token) rather than as a crash at token 8000.
        #
        # Per token, for this model on a 16 GB card:
        #   KV cache       2 * n_kv_heads * head_dim * 2 B (fp16) * full-attention layers
        #                  = 2*4*256*2*16 = 64 KB/token -- the 48 GDN layers hold fixed-size SSM
        #                  state and contribute nothing here
        #   attention ws   n_heads * ATTN_CHUNK * 4 B = 24*256*4 = 24 KB/token, shared across blocks
        # plus a fixed part: the activations (bounded by Network.prefill_rows), the host-weight
        # staging slot, and the dequant scratch.
        #
        # Measured at 16384 tokens: 1073 MB KV + 402 MB workspace + ~300 MB fixed = ~1775 MB, against
        # the 1650 MB that a 52-layer placement actually leaves free. Hence the OOM, and hence placing
        # fewer layers. The fragmentation margin is not cosmetic: with buffers growing and being freed
        # as context extends, a 32 MB allocation was observed failing with 333 MB free.
        max_context = (ENV["SHAINET_MAX_CONTEXT"]? || "16384").to_i
        full_attn_layers = types.count { |t| t != "linear_attention" }
        kv_elem_bytes = ENV.fetch("SHAINET_KV_FP16", "1") == "0" ? 4 : 2
        kv_per_token = 2_i64 * n_kv_heads * head_dim * kv_elem_bytes * full_attn_layers
        # The attention workspace term depends on which attention path runs. The single-block kernel
        # stages a full score row per query token, n_heads * ATTN_CHUNK * 4 B = 24 KB per token of
        # context, which grows without bound.
        #
        # The split-KV path is held under a FIXED budget instead (ATTN_SPLIT_WS_BUDGET_FLOATS, 64 MB)
        # by capping the query chunk as the split count rises, so it contributes a constant rather
        # than a per-token term. That is what makes the reserve winnable: while the workspace grew
        # with context, every megabyte added to the reserve was absorbed by the same growth, and a
        # full-context session failed at ~15979 tokens regardless.
        split_attn = ENV.fetch("SHAINET_SPLIT_ATTN", "1") != "0" && CUDA.attention_split_kv_available?
        ws_per_token = split_attn ? 0_i64 : n_heads.to_i64 * 256 * 4
        ws_fixed = split_attn ? 64_i64 * 1024 * 1024 : 0_i64
        # The fixed term is not a fudge: per-layer SSM state is 3.1 MB (48 heads x 128 x 128 x 4 B)
        # and there are ~39 GDN layers on the device, the SwiGLU trio at Network.prefill_rows is
        # ~107 MB, the host-weight staging slot is up to 70 MB, the dequant scratch 20 MB, and the
        # conv state and resident mixer buffers add more. The 20% on top covers the transient peak
        # when a growing buffer holds its old and new allocation at once.
        #
        # Sized from real failures, not a model. The history is worth keeping because each step was
        # driven by an observed OOM rather than a calculation:
        #
        #   768 MB  -> a reported session died at 15401 of 16384 tokens, 64 MB alloc, 79 MB free.
        #   1024 MB -> still died, at 15979 tokens with 104 MB free.
        #   1280 MB -> still died at 15979 with 118 MB free. The +308 MB of reserve bought +14 MB of
        #              headroom, which is the tell: the failing allocation GREW with context, so the
        #              reserve could never catch it. Fixed by bounding the attention workspace
        #              instead (ATTN_SPLIT_WS_BUDGET_FLOATS), which is the real repair.
        #
        # Back to 896 MB now that the workspace contributes a constant: the reserve no longer has to
        # outrun a growing term, and over-reserving costs layers on every run.
        fixed_bytes = 896_i64 * 1024 * 1024
        derived = ((kv_per_token + ws_per_token) * max_context + fixed_bytes + ws_fixed)
        derived = (derived * 120) // 100
        derived_mb = (derived // (1024 * 1024)).to_i

        # An explicit reserve always wins: it is the escape hatch for trading context for speed.
        reserve_mb = if env = ENV["SHAINET_GGUF_RESERVE_MB"]?
                       env.to_i
                     else
                       derived_mb
                     end
        kv_mb = (kv_per_token * max_context) // (1024 * 1024)
        ws_mb = (ws_per_token * max_context + ws_fixed) // (1024 * 1024)
        Log.info do
          "gguf: reserve #{reserve_mb} MB for #{max_context}-token context " \
          "(KV #{kv_mb} MB + workspace #{ws_mb} MB + fixed/margin)"
        end
        if CUDA.fully_available?
          if info = CUDA.memory_info
            budget = (info[:free].to_i64) - (reserve_mb.to_i64 * 1024 * 1024)
            # lm_head is always placed on the device, so it comes out of the budget first.
            budget -= lm_head_bytes.to_i64
            fits = 0
            (num_transformer_layers - 1).downto(0) do |i|
              break if budget < layer_bytes[i].to_i64
              budget -= layer_bytes[i].to_i64
              fits += 1
            end
            gpu_layers = fits
          else
            # No usable reading: placing everything on the device is the one choice that cannot
            # be right for a model larger than any single card, and it fails as an OOM mid-load
            # rather than as a slow run. Keep the whole trunk on the host instead.
            Log.warn { "gguf: CUDA.memory_info unavailable; keeping all layers on the host" }
            gpu_layers = 0
          end
        end
        cpu_layers = num_transformer_layers - gpu_layers
        placed_mb = (layer_bytes[cpu_layers, gpu_layers].sum(0_u64) + (gpu_layers > 0 ? lm_head_bytes : 0_u64)) // (1024 * 1024)
        Log.info { "gguf: #{gpu_layers}/#{num_transformer_layers} layers on GPU (#{placed_mb} MB placed), #{cpu_layers} on CPU (reserve #{reserve_mb} MB)" }

        # Bulk CUDA allocation: one malloc + one memcpy for ALL GPU tensors.
        # This replaces 416 individual malloc+memcpy calls (was 292s, Ollama does 7.58s).
        # Since GGUF tensors are contiguous in the file, we find the byte range covering
        # all GPU-layer tensors + lm_head and copy it in one shot.
        gpu_pool_ptr = Pointer(UInt8).null
        gpu_pool_map = Hash(UInt64, UInt64).new
        if gpu_layers > 0 && CUDA.fully_available? && gf.mmap?
          # Collect all tensor byte ranges for GPU layers
          gpu_tensor_ranges = [] of {UInt64, UInt64} # {offset_in_tensor_data, byte_size}
          types.each_with_index do |_t, idx|
            next unless idx >= cpu_layers
            gf.tensors.each do |name, info|
              if name.starts_with?("blk.#{idx}.")
                gpu_tensor_ranges << {info.offset, info.byte_size}
              end
            end
          end
          # lm_head
          lm_info = gf.tensors["output.weight"]
          gpu_tensor_ranges << {lm_info.offset, lm_info.byte_size}

          if gpu_tensor_ranges.size > 0
            pool_size = gpu_tensor_ranges.sum(&.[1])
            t0 = Time.monotonic
            CUDA.malloc(pointerof(gpu_pool_ptr).as(Pointer(Pointer(Void))), pool_size)
            # Copy each tensor's data from mmap to the pool, packing them contiguously.
            # Build an offset map so load_gguf_weight_device can find each tensor in the pool.
            pool_offset = 0_u64
            gpu_tensor_ranges.sort_by!(&.[0])
            gpu_tensor_ranges.each do |(file_off, sz)|
              if src = gf.tensor_ptr(GGUF::TensorInfo.new("_", [0_u64], GGUF::GGMLType::F32, file_off))
                CUDA.memcpy((gpu_pool_ptr + pool_offset).as(Pointer(Void)), src.as(Pointer(Void)),
                  sz, CUDA::MemcpyKind::HostToDevice)
              end
              # Record the mapping: file_offset -> pool_offset
              gpu_pool_map[file_off] = pool_offset
              pool_offset += sz
            end
            dt = (Time.monotonic - t0).total_seconds
            Log.info { "gguf: bulk GPU alloc #{pool_size / (1024*1024)} MB in #{dt.round(2)}s (#{gpu_tensor_ranges.size} tensors)" }
          end
        end

        types.each_with_index do |t, idx|
          on_gpu = idx >= cpu_layers # back-to-front: last layers go on GPU first
          if t == "linear_attention"
            load_gguf_linear_attn_layer(gf, net, idx, d, ff, eps, ssm_heads, ssm_k_heads,
              ssm_head_dim, conv_kernel, ssm_state, on_gpu, gpu_pool_ptr, gpu_pool_map)
          else
            load_gguf_full_attn_layer(gf, net, idx, d, ff, eps, n_heads, n_kv_heads,
              head_dim, rope_theta, partial_rotary, on_gpu, gpu_pool_ptr, gpu_pool_map)
          end
          GC.collect unless on_gpu
          @@progress.try &.call(idx + 1, num_transformer_layers)
        end

        # Output layer
        net.add_layer(:output, 248320, activation_function: SHAInet.identity)
        net.fully_connect
        output_layer = net.output_layers.first
        output_layer.biases = SimpleMatrix.new(1, 248320)

        # Final norm
        fn_info = gf.tensors["output_norm.weight"]
        fn_data = read_gguf_f32_tensor(gf, fn_info)
        final_norm = RMSNorm.new(d, eps)
        gamma = SimpleMatrix.new(1, d)
        d.times { |i| gamma[0, i] = fn_data[i].to_f32 } # GGUF stores final gamma
        final_norm.gamma = gamma
        net.final_norm = final_norm

        # lm_head
        lm_info = gf.tensors["output.weight"]
        net.lm_head_q = load_gguf_weight_device(gf, lm_info, gpu_pool_ptr, gpu_pool_map)
        net.quantize_weights = true

        emb_layer.to_host! if ENV.fetch("SHAINET_EMBED_HOST", "1") != "0"
        # Pin every attention block's KV cache to the context the reserve was sized for, so the cache
        # is allocated ONCE at that size instead of doubling into it.
        #
        # The doubling was the proximate cause of the mid-conversation OOMs, not raw capacity: each
        # growth allocates the new buffer, copies the old one across device-to-device, and frees the
        # old, which leaves holes. Observed on a growing conversation, a 32 MB allocation failed with
        # 333 MB free, and the per-step VRAM deltas swung between +740 MB and -95 MB. With the cache
        # pinned, the same run completes a full 16016 tokens.
        #
        # It also converts an OOM at token 8000 into an ArgumentError at the point the budget is
        # exceeded, which a caller can act on.
        net.layers.each do |layer|
          layer.kv_max_context = max_context if layer.responds_to?(:kv_max_context=)
        end

        Log.info { "gguf: loaded #{num_transformer_layers} layers from #{path}" }
        net
        # NOTE: gf is NOT closed here. The mmap must stay alive for the model's lifetime
        # because the Q4_K embedding pointer and GGUFHostMatrix pointers point directly
        # into the mmap'd region. Closing it would munmap and SIGSEGV on first access.
        # The OS reclaims the mapping when the process exits.
      end
    end

    # Load a GGUF tensor as either GGUFMatrix (device) or SimpleMatrix (host dequanted).
    private def self.load_gguf_weight(gf : GGUF::File, info : GGUF::TensorInfo, on_gpu : Bool = true,
                                      gpu_pool : Pointer(UInt8) = Pointer(UInt8).null,
                                      pool_map : Hash(UInt64, UInt64) = Hash(UInt64, UInt64).new) : QuantizedWeight | SimpleMatrix | CudaMatrix
      unless on_gpu
        if mmap_ptr = gf.tensor_ptr(info)
          rows = info.shape[0].to_i32
          cols = info.shape.size > 1 ? info.shape[1].to_i32 : 1
          return GGUFHostMatrix.new(rows, cols, info.type, mmap_ptr, info.byte_size)
        end
      end
      load_gguf_weight_device(gf, info, gpu_pool, pool_map)
    end

    private def self.load_gguf_weight_device(gf : GGUF::File, info : GGUF::TensorInfo,
                                             gpu_pool : Pointer(UInt8) = Pointer(UInt8).null,
                                             pool_map : Hash(UInt64, UInt64) = Hash(UInt64, UInt64).new) : GGUFMatrix
      # GGUF shape: ne0 = fastest dim (values per row = K), ne1 = rows (N)
      rows = info.shape[0].to_i32
      cols = info.shape.size > 1 ? info.shape[1].to_i32 : 1
      # Use the bulk pool if available (data already on device from one big memcpy)
      if !gpu_pool.null? && (pool_off = pool_map[info.offset]?)
        dev_ptr = gpu_pool + pool_off
        return GGUFMatrix.from_pool(rows, cols, info.type, dev_ptr, info.byte_size)
      end
      # Fallback: individual malloc+memcpy
      if mmap_ptr = gf.tensor_ptr(info)
        GGUFMatrix.new(rows, cols, info.type, mmap_ptr, info.byte_size)
      else
        ptr = gf.read_tensor_raw(info)
        GGUFMatrix.new(rows, cols, info.type, ptr, info.byte_size)
      end
    end

    # Read a GGUF F32 tensor into a flat Array(Float32).
    private def self.read_gguf_f32_tensor(gf : GGUF::File, info : GGUF::TensorInfo) : Array(Float32)
      count = info.element_count.to_i32
      if info.type == GGUF::GGMLType::F32
        if ptr = gf.tensor_ptr(info)
          Array(Float32).new(count) { |i| (ptr + i * 4).as(Pointer(Float32)).value }
        else
          buf = Bytes.new(count * 4)
          gf.read_tensor_data(info, buf)
          Array(Float32).new(count) { |i| IO::ByteFormat::LittleEndian.decode(Float32, buf[i * 4, 4]) }
        end
      elsif info.type == GGUF::GGMLType::Q4_K
        dequant_q4k_host(gf, info)
      elsif info.type == GGUF::GGMLType::Q6_K
        dequant_q6k_host(gf, info)
      else
        raise "read_gguf_f32_tensor: unsupported type #{info.type} for #{info.name}"
      end
    end

    # Host-side Q4_K dequantization (for small tensors only -- embedding lookup, alpha/beta).
    private def self.dequant_q4k_host(gf : GGUF::File, info : GGUF::TensorInfo) : Array(Float32)
      if ptr = gf.tensor_ptr(info)
        raw_ptr = ptr
      else
        raw = Bytes.new(info.byte_size.to_i32)
        gf.read_tensor_data(info, raw)
        raw_ptr = raw.to_unsafe
      end
      count = info.element_count.to_i32
      result = Array(Float32).new(count, 0.0_f32)
      nblocks = count // 256
      nblocks.times do |blk|
        block = raw_ptr + blk * 144
        d = half_to_f32(block[0].to_u16 | (block[1].to_u16 << 8))
        dmin = half_to_f32(block[2].to_u16 | (block[3].to_u16 << 8))
        scales = block + 4
        qs = block + 16
        base = blk * 256
        4.times do |j64|
          sc0, m0 = get_scale_min_k4_host(j64 * 2, scales)
          sc1, m1 = get_scale_min_k4_host(j64 * 2 + 1, scales)
          d1 = d * sc0.to_f32
          m1_val = dmin * m0.to_f32
          d2 = d * sc1.to_f32
          m2_val = dmin * m1.to_f32
          32.times do |l|
            result[base + j64 * 64 + l] = d1 * (qs[j64 * 32 + l] & 0xF).to_f32 - m1_val
            result[base + j64 * 64 + l + 32] = d2 * (qs[j64 * 32 + l] >> 4).to_f32 - m2_val
          end
        end
      end
      result
    end

    # Host-side Q6_K dequantization.
    private def self.dequant_q6k_host(gf : GGUF::File, info : GGUF::TensorInfo) : Array(Float32)
      if ptr = gf.tensor_ptr(info)
        raw_ptr = ptr
      else
        raw = Bytes.new(info.byte_size.to_i32)
        gf.read_tensor_data(info, raw)
        raw_ptr = raw.to_unsafe
      end
      count = info.element_count.to_i32
      result = Array(Float32).new(count, 0.0_f32)
      nblocks = count // 256
      nblocks.times do |blk|
        block = raw_ptr + blk * 210
        ql = block
        qh = block + 128
        sc = block + 192
        d = half_to_f32(block[208].to_u16 | (block[209].to_u16 << 8))
        base = blk * 256
        (256 // 128).times do |chunk|
          ql_c = ql + chunk * 64
          qh_c = qh + chunk * 32
          sc_c = sc + chunk * 8
          32.times do |l|
            is = l // 16
            q1 = ((ql_c[l] & 0xF) | (((qh_c[l] >> 0) & 3) << 4)).to_i8 - 32
            q2 = ((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)).to_i8 - 32
            q3 = ((ql_c[l] >> 4) | (((qh_c[l] >> 4) & 3) << 4)).to_i8 - 32
            q4 = ((ql_c[l + 32] >> 4) | (((qh_c[l] >> 6) & 3) << 4)).to_i8 - 32
            result[base + chunk * 128 + l] = d * sc_c[is].to_i8!.to_f32 * q1.to_f32
            result[base + chunk * 128 + l + 32] = d * sc_c[is + 2].to_i8!.to_f32 * q2.to_f32
            result[base + chunk * 128 + l + 64] = d * sc_c[is + 4].to_i8!.to_f32 * q3.to_f32
            result[base + chunk * 128 + l + 96] = d * sc_c[is + 6].to_i8!.to_f32 * q4.to_f32
          end
        end
      end
      result
    end

    def self.get_scale_min_k4_host(j : Int32, scales : Pointer(UInt8)) : {UInt8, UInt8}
      if j < 4
        {scales[j] & 63_u8, scales[j + 4] & 63_u8}
      else
        sc = (scales[j + 4] & 0xF_u8) | ((scales[j - 4] >> 6) << 4)
        mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
        {sc, mn}
      end
    end

    def self.half_to_f32(bits : UInt16) : Float32
      sign = (bits >> 15) & 1
      exp = (bits >> 10) & 0x1F
      frac = bits & 0x3FF
      if exp == 0
        return sign == 1 ? -0.0_f32 : 0.0_f32 if frac == 0
        # Subnormal
        val = frac.to_f32 / 1024.0_f32 * (2.0_f32 ** -14)
        return sign == 1 ? -val : val
      elsif exp == 31
        return sign == 1 ? Float32::INFINITY * -1 : Float32::INFINITY if frac == 0
        return Float32::NAN
      end
      val = (1.0_f32 + frac.to_f32 / 1024.0_f32) * (2.0_f32 ** (exp.to_i32 - 15))
      sign == 1 ? -val : val
    end

    # Read a GGUF F32 tensor into a SimpleMatrix.
    # GGUF flat data: ne1 groups of ne0 values (ne0 fastest).
    # For a weight [ne0, ne1] used as y = x @ W where W is [rows=ne0, cols=ne1],
    # the flat data has cols groups of rows values.
    private def self.read_gguf_f32_matrix(gf : GGUF::File, info : GGUF::TensorInfo, rows : Int32, cols : Int32) : SimpleMatrix
      data = read_gguf_f32_tensor(gf, info)
      m = SimpleMatrix.new(rows, cols)
      cols.times { |c| rows.times { |r| m[r, c] = data[c * rows + r].to_f64 } }
      m
    end

    private def self.load_gguf_embedding(gf : GGUF::File, info : GGUF::TensorInfo,
                                         emb_layer : EmbeddingLayer, d : Int32)
      vocab = info.shape[1].to_i32
      if (info.type == GGUF::GGMLType::Q4_K || info.type == GGUF::GGMLType::Q6_K) && (ptr = gf.tensor_ptr(info))
        # Keep the Q4_K data mmap'd and dequant per-row during embed_cpu.
        # This turns a 5-minute full dequant into a ~0.01 ms per-row lookup.
        emb_layer.q4k_embedding = {ptr, info.byte_size, vocab, d}
        Log.info { "gguf: embedding kept as Q4_K (#{vocab} x #{d}), per-row dequant" }
      else
        # F32 or no mmap: full dequant to fp32
        data = read_gguf_f32_tensor(gf, info)
        host_emb = SimpleMatrix.new(vocab, d)
        vocab.times { |v| d.times { |j| host_emb[v, j] = data[v * d + j].to_f64 } }
        emb_layer.embeddings = host_emb
      end
    end

    private def self.load_gguf_linear_attn_layer(gf : GGUF::File, net : Network, idx : Int32,
                                                 d : Int32, ff : Int32, eps : Float64,
                                                 num_v_heads : Int32, num_k_heads : Int32,
                                                 head_dim : Int32, conv_kernel : Int32,
                                                 state_size : Int32, on_gpu : Bool,
                                                 gpu_pool : Pointer(UInt8) = Pointer(UInt8).null,
                                                 pool_map : Hash(UInt64, UInt64) = Hash(UInt64, UInt64).new)
      net.add_layer("gated_deltanet", d, allocate: false, num_heads: num_v_heads,
        ff_hidden: ff, num_kv_heads: num_k_heads,
        eps: eps, head_dim: head_dim,
        linear_conv_kernel: conv_kernel)
      block = net.hidden_layers.last.as(GatedDeltaNetBlock)
      # Fused QKV: blk.N.attn_qkv.weight [d, q_out + k_out + v_out]
      # Split into separate Q, K, V by creating sub-views into the device buffer.
      qkv_info = gf.tensors["blk.#{idx}.attn_qkv.weight"]
      k_dim = num_k_heads * head_dim
      v_dim = num_v_heads * head_dim
      q_out = k_dim                     # Q output dim = k_dim (key-dimension queries)
      k_out = k_dim                     # K output dim = k_dim
      v_out = v_dim                     # V output dim = v_dim
      in_dim = qkv_info.shape[0].to_i32 # d
      q_w, k_w, v_w = split_gguf_qkv(gf, qkv_info, in_dim, q_out, k_out, v_out,
        on_gpu, gpu_pool, pool_map)
      block.w_q = q_w
      block.w_k = k_w
      block.w_v = v_w

      # Gate (z projection): blk.N.attn_gate.weight [d, gate_dim]
      gate_info = gf.tensors["blk.#{idx}.attn_gate.weight"]
      block.w_gate = load_gguf_weight(gf, gate_info, on_gpu, gpu_pool, pool_map)

      # Output: blk.N.ssm_out.weight [ssm_inner, d]
      out_info = gf.tensors["blk.#{idx}.ssm_out.weight"]
      block.w_o = load_gguf_weight(gf, out_info, on_gpu, gpu_pool, pool_map)

      # Alpha/Beta: dequant to fp32 (small matrices, block expects SimpleMatrix | CudaMatrix)
      alpha_info = gf.tensors["blk.#{idx}.ssm_alpha.weight"]
      beta_info = gf.tensors["blk.#{idx}.ssm_beta.weight"]
      alpha_r = alpha_info.shape[0].to_i32
      alpha_c = alpha_info.shape.size > 1 ? alpha_info.shape[1].to_i32 : 1
      block.w_alpha = read_gguf_f32_matrix(gf, alpha_info, alpha_r, alpha_c)
      block.w_beta = read_gguf_f32_matrix(gf, beta_info, alpha_r, alpha_c)
      block.w_alpha = block.w_alpha.as(SimpleMatrix).to_cuda if CUDA.fully_available?
      block.w_beta = block.w_beta.as(SimpleMatrix).to_cuda if CUDA.fully_available?

      # GGUF lays the fused projection out flat as [q * num_k_heads, k * num_k_heads,
      # v * num_v_heads]; llama.cpp then widens q/k to num_v_heads with ggml_repeat, which TILES.
      # So value head h reads key head h % num_k_heads, not h // heads_per_k as the SafeTensors
      # layout wants. Verified elementwise against llama.cpp's dumped attn_output (cosine and
      # magnitude ratio both 1.000000 at layers 0, 10 and 21).
      block.k_head_tiled = true

      # A_log and dt_bias (tiny F32 vectors)
      #
      # GGUF's `ssm_a` is NOT the raw HF `A_log`: llama.cpp's converter stores the
      # pre-computed `-exp(A_log)` (always negative, e.g. -0.04). llama.cpp then uses it
      # directly as `alpha = exp(softplus(proj + dt_bias) * ssm_a)`.
      #
      # GatedDeltaNetBlock#gates expects the HF form and computes
      # `alpha = exp(-exp(a_log) * softplus(proj + dt_bias))`, so feed it
      # `a_log = log(-ssm_a)` to make `-exp(a_log)` equal the stored `ssm_a`.
      #
      # Reading `ssm_a` as if it were A_log applied exp() to an already-exponentiated
      # value: for ssm_a = -0.0406 that yields a decay exponent of -0.96 instead of
      # -0.0406, ~24x too much decay in every one of the 48 linear-attention layers.
      a_log_info = gf.tensors["blk.#{idx}.ssm_a"]
      dt_info = gf.tensors["blk.#{idx}.ssm_dt.bias"]
      a_log = read_gguf_f32_tensor(gf, a_log_info)
      dt_bias = read_gguf_f32_tensor(gf, dt_info)
      num_v_heads.times do |h|
        neg_a = -a_log[h].to_f64
        raise "gguf: ssm_a[#{h}] of blk.#{idx} is #{a_log[h]}, expected negative (-exp(A_log))" unless neg_a > 0.0
        block.a_log[h] = Math.log(neg_a)
        block.dt_bias[h] = dt_bias[h].to_f64
      end

      # Conv1d (F32)
      conv_info = gf.tensors["blk.#{idx}.ssm_conv1d.weight"]
      conv_data = read_gguf_f32_tensor(gf, conv_info)
      k_dim = num_k_heads * head_dim
      v_dim = num_v_heads * head_dim
      # Conv shape: GGUF [conv_kernel, total_ch] with ne0=conv_kernel fastest.
      # Data: ne1=total_ch groups of ne0=conv_kernel values.
      total_ch = 2 * k_dim + v_dim
      conv_m = SimpleMatrix.new(total_ch, conv_kernel)
      total_ch.times { |c| conv_kernel.times { |t| conv_m[c, t] = conv_data[t + c * conv_kernel].to_f64 } }
      # Split into q, k, v and reverse taps
      cq = SimpleMatrix.new(k_dim, conv_kernel)
      ck = SimpleMatrix.new(k_dim, conv_kernel)
      cv = SimpleMatrix.new(v_dim, conv_kernel)
      k_dim.times { |i| conv_kernel.times { |t| cq[i, t] = conv_m[i, t] } }
      k_dim.times { |i| conv_kernel.times { |t| ck[i, t] = conv_m[k_dim + i, t] } }
      v_dim.times { |i| conv_kernel.times { |t| cv[i, t] = conv_m[2 * k_dim + i, t] } }
      reverse_taps!(cq, block.conv_q.weight)
      reverse_taps!(ck, block.conv_k.weight)
      reverse_taps!(cv, block.conv_v.weight)

      # Output norm (SSM norm)
      norm_info = gf.tensors["blk.#{idx}.ssm_norm.weight"]
      block.out_norm.gamma = read_gguf_f32_matrix(gf, norm_info, 1, state_size)

      # Layer norms (GGUF stores final gamma = 1 + offset, not the raw offset)
      n1_info = gf.tensors["blk.#{idx}.attn_norm.weight"]
      n2_info = gf.tensors["blk.#{idx}.post_attention_norm.weight"]
      block.norm1.gamma = read_gguf_f32_matrix(gf, n1_info, 1, d)
      block.norm2.gamma = read_gguf_f32_matrix(gf, n2_info, 1, d)
      block.norm1.to_gpu!
      block.norm2.to_gpu!
      block.out_norm.to_gpu!

      # FFN
      ffn = block.ffn
      ffn.gate_proj = load_gguf_weight(gf, gf.tensors["blk.#{idx}.ffn_gate.weight"], on_gpu, gpu_pool, pool_map)
      ffn.up_proj = load_gguf_weight(gf, gf.tensors["blk.#{idx}.ffn_up.weight"], on_gpu, gpu_pool, pool_map)
      ffn.down_proj = load_gguf_weight(gf, gf.tensors["blk.#{idx}.ffn_down.weight"], on_gpu, gpu_pool, pool_map)
    end

    # Split the interleaved Q+gate projection WITHOUT dequantizing it.
    #
    # attn_q.weight is [ne0 = in_dim, ne1 = n_heads * head_dim * 2] and the Q/gate interleaving runs
    # along ne1 -- the OUTPUT rows -- while k-quant blocks run along ne0. Every output row is
    # therefore quantized independently, so the split is a copy of whole raw rows, not a numeric
    # operation, and the result is still Q4_K/Q6_K.
    #
    # Dequantizing to fp32 first, which is what this replaced, cost 120 MB per matrix and 240 MB per
    # layer: 3.84 GB across the 16 full-attention layers, for weights that are 17.7 MB each
    # quantized. It also left both projections on the HOST as plain SimpleMatrix, which is why a
    # full-attention layer measured 13.8 ms per decode token against 3.98 ms for a device-resident
    # Gated DeltaNet layer.
    #
    # Destination layout matches HFLoader#split_head_interleaved: output row h*head_dim + i comes
    # from source row h*2*head_dim + i for Q, and + head_dim for the gate.
    private def self.split_gguf_q_gate(gf : GGUF::File, info : GGUF::TensorInfo,
                                       n_heads : Int32, head_dim : Int32, on_gpu : Bool)
      in_dim = info.shape[0].to_i32
      out_total = info.shape[1].to_i32
      half = n_heads * head_dim
      raise "split_gguf_q_gate: #{out_total} outputs is not 2 * #{half}" unless out_total == half * 2

      bs, vs = GGUF::BLOCK_SIZE[info.type]? || raise "unsupported type #{info.type} for Q/gate split"
      row_bytes = ((in_dim + vs - 1) // vs) * bs
      src = gf.tensor_ptr(info) || raise "attn_q tensor has no data pointer"

      total = half.to_u64 * row_bytes
      qbuf = Pointer(UInt8).malloc(total)
      gbuf = Pointer(UInt8).malloc(total)
      n_heads.times do |h|
        head_dim.times do |i|
          dst = (h * head_dim + i).to_u64 * row_bytes
          q_off = (h * 2 * head_dim + i).to_u64 * row_bytes
          g_off = (h * 2 * head_dim + head_dim + i).to_u64 * row_bytes
          (qbuf + dst).copy_from(src + q_off, row_bytes)
          (gbuf + dst).copy_from(src + g_off, row_bytes)
        end
      end

      if on_gpu && CUDA.fully_available?
        {GGUFMatrix.new(in_dim, half, info.type, qbuf, total),
         GGUFMatrix.new(in_dim, half, info.type, gbuf, total)}
      else
        # The buffers are GC-allocated rather than mmap-backed, and GGUFHostMatrix keeps only the
        # raw pointer, so hand it the owning slice as well to pin them for the model's lifetime.
        {GGUFHostMatrix.new(in_dim, half, info.type, qbuf, total, Bytes.new(qbuf, total)),
         GGUFHostMatrix.new(in_dim, half, info.type, gbuf, total, Bytes.new(gbuf, total))}
      end
    end

    private def self.load_gguf_full_attn_layer(gf : GGUF::File, net : Network, idx : Int32,
                                               d : Int32, ff : Int32, eps : Float64,
                                               n_heads : Int32, n_kv_heads : Int32,
                                               head_dim : Int32, rope_theta : Float64,
                                               partial_rotary : Int32, on_gpu : Bool,
                                               gpu_pool : Pointer(UInt8) = Pointer(UInt8).null,
                                               pool_map : Hash(UInt64, UInt64) = Hash(UInt64, UInt64).new)
      net.add_layer(:llama, d, allocate: false, num_heads: n_heads, ff_hidden: ff,
        num_kv_heads: n_kv_heads, eps: eps, head_dim: head_dim)
      block = net.hidden_layers.last.as(LlamaBlock)
      block.rope_theta = rope_theta
      block.rotary_dim = partial_rotary if partial_rotary < head_dim

      # Separate Q/K/V/O for full attention layers
      # Q weight may be [d, q_dim*2] with interleaved Q+gate per head.
      q_info = gf.tensors["blk.#{idx}.attn_q.weight"]
      q_out = q_info.shape[1].to_i32
      q_dim = n_heads * head_dim
      if q_out == q_dim * 2
        wq, wg = split_gguf_q_gate(gf, q_info, n_heads, head_dim, on_gpu)
        block.w_q = wq
        block.w_gate_attn = wg
      else
        block.w_q = load_gguf_weight(gf, q_info, on_gpu, gpu_pool, pool_map)
      end
      block.w_k = load_gguf_weight(gf, gf.tensors["blk.#{idx}.attn_k.weight"], on_gpu, gpu_pool, pool_map)
      block.w_v = load_gguf_weight(gf, gf.tensors["blk.#{idx}.attn_v.weight"], on_gpu, gpu_pool, pool_map)
      block.w_o = load_gguf_weight(gf, gf.tensors["blk.#{idx}.attn_output.weight"], on_gpu, gpu_pool, pool_map)

      # Q/K norms (offset-from-one for qwen3.5 RMSNorm)
      qn_data = read_gguf_f32_tensor(gf, gf.tensors["blk.#{idx}.attn_q_norm.weight"])
      kn_data = read_gguf_f32_tensor(gf, gf.tensors["blk.#{idx}.attn_k_norm.weight"])
      block.q_norm = Array(Float32).new(qn_data.size) { |i| qn_data[i] }
      block.k_norm = Array(Float32).new(kn_data.size) { |i| kn_data[i] }

      # Layer norms (GGUF stores final gamma, not offset)
      n1_info = gf.tensors["blk.#{idx}.attn_norm.weight"]
      n2_info = gf.tensors["blk.#{idx}.post_attention_norm.weight"]
      block.norm1.gamma = read_gguf_f32_matrix(gf, n1_info, 1, d)
      block.norm2.gamma = read_gguf_f32_matrix(gf, n2_info, 1, d)
      block.norm1.to_gpu!
      block.norm2.to_gpu!

      # FFN
      ffn = block.ffn.as(SwiGLUFF)
      ffn.gate_proj = load_gguf_weight(gf, gf.tensors["blk.#{idx}.ffn_gate.weight"], on_gpu, gpu_pool, pool_map)
      ffn.up_proj = load_gguf_weight(gf, gf.tensors["blk.#{idx}.ffn_up.weight"], on_gpu, gpu_pool, pool_map)
      ffn.down_proj = load_gguf_weight(gf, gf.tensors["blk.#{idx}.ffn_down.weight"], on_gpu, gpu_pool, pool_map)
    end
  end
end

module SHAInet
  module HFLoader
    # Dequant a single row from a Q4_K embedding table.
    # The embedding is stored as [vocab, d] with each row of d values packed into
    # ceil(d/256) Q4_K blocks (144 bytes each). This dequants one row (one token)
    # and writes d fp32 values into dst starting at dst_offset.
    def self.dequant_q4k_row(base_ptr : Pointer(UInt8), row : Int32, d : Int32,
                             dst : Array(Float32), dst_offset : Int32)
      blocks_per_row = (d + 255) // 256
      row_bytes = blocks_per_row * 144
      row_ptr = base_ptr + row.to_i64 * row_bytes

      col = 0
      blocks_per_row.times do |blk|
        block = row_ptr + blk * 144
        d_scale = half_to_f32((block[0].to_u16 | (block[1].to_u16 << 8)))
        dmin = half_to_f32((block[2].to_u16 | (block[3].to_u16 << 8)))
        scales = block + 4
        qs = block + 16

        4.times do |j64|
          sc0, m0 = get_scale_min_k4_host(j64 * 2, scales)
          sc1, m1 = get_scale_min_k4_host(j64 * 2 + 1, scales)
          d1 = d_scale * sc0.to_f32
          m1_val = dmin * m0.to_f32
          d2 = d_scale * sc1.to_f32
          m2_val = dmin * m1.to_f32
          32.times do |l|
            idx = col + j64 * 64 + l
            break if idx >= d
            dst[dst_offset + idx] = (d1 * (qs[j64 * 32 + l] & 0xF).to_f32 - m1_val).to_f32
          end
          32.times do |l|
            idx = col + j64 * 64 + l + 32
            break if idx >= d
            dst[dst_offset + idx] = (d2 * (qs[j64 * 32 + l] >> 4).to_f32 - m2_val).to_f32
          end
        end
        col += 256
      end
    end
  end
end

module SHAInet
  module HFLoader
    # Split a fused QKV tensor into separate Q, K, V weight matrices.
    # The fused tensor [in_dim, q_out + k_out + v_out] is stored as rows of k-quant blocks.
    # Each output neuron is one row. Q occupies the first q_out rows, then K, then V.
    private def self.split_gguf_qkv(gf : GGUF::File, info : GGUF::TensorInfo,
                                    in_dim : Int32, q_out : Int32, k_out : Int32, v_out : Int32,
                                    on_gpu : Bool,
                                    gpu_pool : Pointer(UInt8),
                                    pool_map : Hash(UInt64, UInt64)) : {QuantizedWeight | SimpleMatrix | CudaMatrix, QuantizedWeight | SimpleMatrix | CudaMatrix, QuantizedWeight | SimpleMatrix | CudaMatrix}
      bs, vs = GGUF::BLOCK_SIZE[info.type]
      blocks_per_row = (in_dim.to_u64 + vs.to_u64 - 1) // vs.to_u64
      bytes_per_row = blocks_per_row * bs.to_u64

      q_bytes = q_out.to_u64 * bytes_per_row
      k_bytes = k_out.to_u64 * bytes_per_row
      v_bytes = v_out.to_u64 * bytes_per_row

      if !gpu_pool.null? && (pool_off = pool_map[info.offset]?)
        # Device pool: create sub-views
        base = gpu_pool + pool_off
        q = GGUFMatrix.from_pool(in_dim, q_out, info.type, base, q_bytes)
        k = GGUFMatrix.from_pool(in_dim, k_out, info.type, base + q_bytes, k_bytes)
        v = GGUFMatrix.from_pool(in_dim, v_out, info.type, base + q_bytes + k_bytes, v_bytes)
        {q, k, v}
      elsif !on_gpu && (mmap_ptr = gf.tensor_ptr(info))
        # Host mmap: sub-views
        q = GGUFHostMatrix.new(in_dim, q_out, info.type, mmap_ptr, q_bytes)
        k = GGUFHostMatrix.new(in_dim, k_out, info.type, mmap_ptr + q_bytes, k_bytes)
        v = GGUFHostMatrix.new(in_dim, v_out, info.type, mmap_ptr + q_bytes + k_bytes, v_bytes)
        {q, k, v}
      else
        # Fallback: load the whole tensor and split
        raise "split_gguf_qkv: no pool or mmap available"
      end
    end
  end
end
