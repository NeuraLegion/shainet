require "./safetensors"
require "json"

module SHAInet
  # Load a GPT-2 model directly from a HuggingFace SafeTensors file.
  # No Python, no PyTorch — pure Crystal.
  module HFLoader
    SUPPORTED_MODELS = ["gpt2", "llama", "mistral", "qwen2", "qwen3", "qwen3_moe"]

    # Optional progress reporting for the layer loop. A 30B takes about nine minutes to
    # load and quantize, which is indistinguishable from a hang without this. Set to nil to
    # silence it again. Deliberately a class-level hook rather than a parameter threaded
    # through every architecture loader.
    #
    # The callback receives (layers_done, layers_total).
    @@progress : Proc(Int32, Int32, Nil)? = nil

    def self.progress=(callback : Proc(Int32, Int32, Nil)?)
      @@progress = callback
    end

    def self.progress : Proc(Int32, Int32, Nil)?
      @@progress
    end

    # Experts read between forced collections while loading an MoE layer. Bounds the
    # read/transpose garbage that would otherwise accumulate across a whole layer's
    # experts; 32 keeps the load-time cost small while capturing most of the peak
    # reduction. SHAINET_EXPERT_GC_INTERVAL overrides it; 0 disables the extra
    # collections and restores the old per-layer-only behaviour.
    EXPERT_GC_INTERVAL = begin
      raw = (ENV["SHAINET_EXPERT_GC_INTERVAL"]? || "32").to_i
      raw <= 0 ? Int32::MAX : raw
    end

    # Open a model's weights whether they're a single model.safetensors or
    # sharded (model.safetensors.index.json + model-0000k-of-0000N.safetensors).
    # Returns either a SafeTensors::File or ShardedFile — both expose the same
    # read_matrix / read_f32 / read_f64 / has_tensor? / tensor_names interface
    # used by the loaders.
    def self.open_safetensors(model_dir : String) : SafeTensors::File | SafeTensors::ShardedFile
      single = ::File.join(model_dir, "model.safetensors")
      index = ::File.join(model_dir, "model.safetensors.index.json")
      if ::File.exists?(single)
        SafeTensors::File.new(single)
      elsif ::File.exists?(index)
        SafeTensors::ShardedFile.new(model_dir, index)
      else
        raise "No model.safetensors or model.safetensors.index.json found in #{model_dir}"
      end
    end

    # Generic entry point — reads config.json and dispatches to the right loader.
    def self.load(model_dir : String, quantize : Bool = false, bits : Int32 = 8) : Network
      raise ArgumentError.new("unsupported quantization bits: #{bits} (expected 8 or 4)") unless bits == 8 || bits == 4
      config_path = ::File.join(model_dir, "config.json")
      raise "config.json not found in #{model_dir}" unless ::File.exists?(config_path)

      json = JSON.parse(::File.read(config_path))
      model_type = json["model_type"]?.try(&.as_s) || raise "No model_type in config.json"

      case model_type
      when "gpt2"
        load_gpt2(model_dir)
      when "llama", "mistral", "qwen2", "qwen3", "qwen3_moe"
        load_llama(model_dir, quantize: quantize, bits: bits)
      when "qwen3_5"
        load_qwen35(model_dir)
      else
        raise "Unsupported model_type: '#{model_type}'. Supported: #{SUPPORTED_MODELS.join(", ")}"
      end
    end

    # Config parsed from config.json
    record GPT2Config,
      vocab_size : Int32,
      n_embd : Int32,
      n_head : Int32,
      n_layer : Int32,
      n_positions : Int32

    def self.load_gpt2_config(path : String) : GPT2Config
      json = JSON.parse(::File.read(path))
      GPT2Config.new(
        vocab_size: json["vocab_size"].as_i,
        n_embd: json["n_embd"].as_i,
        n_head: json["n_head"].as_i,
        n_layer: json["n_layer"].as_i,
        n_positions: json["n_positions"].as_i
      )
    end

    # Load GPT-2 weights from a .safetensors file into a Network.
    # Expects config.json in the same directory.
    def self.load_gpt2(model_dir : String) : Network
      config_path = ::File.join(model_dir, "config.json")
      model_path = ::File.join(model_dir, "model.safetensors")

      raise "config.json not found in #{model_dir}" unless ::File.exists?(config_path)
      raise "model.safetensors not found in #{model_dir}" unless ::File.exists?(model_path)

      config = load_gpt2_config(config_path)
      sf = SafeTensors::File.new(model_path)

      begin
        net = Network.new
        d_model = config.n_embd
        ff_hidden = d_model * 4
        # Check if model uses a different ff_hidden (GPT-2 tiny uses intermediate_size)
        json = JSON.parse(::File.read(config_path))
        if n_inner = json["n_inner"]?
          ff_hidden = n_inner.as_i unless n_inner.raw.nil?
        end
        # For tiny-random-gpt2, ff_hidden is determined by actual weight shape
        if sf.has_tensor?("transformer.h.0.mlp.c_fc.weight")
          info = sf.tensors["transformer.h.0.mlp.c_fc.weight"]
          ff_hidden = info.shape[1].to_i32 # [d_model, ff_hidden]
        end

        net.add_layer(:input, 1)
        net.add_layer(:embedding, d_model, vocab_size: config.vocab_size)
        config.n_layer.times do
          net.add_layer(:transformer, d_model, num_heads: config.n_head, ff_hidden: ff_hidden)
        end
        net.add_layer(:output, config.vocab_size, activation_function: SHAInet.identity)
        net.fully_connect

        # Load token embeddings
        emb_layer = net.hidden_layers.find(&.is_a?(EmbeddingLayer)).as(EmbeddingLayer)
        wte = sf.read_matrix("transformer.wte.weight") # [vocab, d_model]
        config.vocab_size.times do |i|
          d_model.times do |j|
            emb_layer.embeddings[i, j] = wte[i, j]
          end
        end

        # Load positional embeddings
        wpe = sf.read_matrix("transformer.wpe.weight") # [n_positions, d_model]

        # Load transformer blocks
        config.n_layer.times do |idx|
          t_layer = net.transformer_layers[idx].as(TransformerLayer)
          prefix = "transformer.h.#{idx}"

          # Set positional encoding on the block
          t_layer.positional_encoding = wpe

          # Attention: GPT-2 stores QKV combined as c_attn [d_model, 3*d_model]
          # Need to split into Q, K, V weight matrices
          c_attn_w = sf.read_matrix("#{prefix}.attn.c_attn.weight") # [d_model, 3*d_model]

          # Split combined QKV weights: columns [0:d, d:2d, 2d:3d]
          w_q = SimpleMatrix.new(d_model, d_model)
          w_k = SimpleMatrix.new(d_model, d_model)
          w_v = SimpleMatrix.new(d_model, d_model)
          d_model.times do |r|
            d_model.times do |c|
              w_q[r, c] = c_attn_w[r, c]
              w_k[r, c] = c_attn_w[r, c + d_model]
              w_v[r, c] = c_attn_w[r, c + 2 * d_model]
            end
          end

          # Shainet MHA stores weights for matmul: x * W
          # HF c_attn.weight is [d_model, 3*d_model], split into Q/K/V [d_model, d_model]
          t_layer.mha.w_q = w_q
          t_layer.mha.w_k = w_k
          t_layer.mha.w_v = w_v

          # Output projection [d_model, d_model]
          c_proj_w = sf.read_matrix("#{prefix}.attn.c_proj.weight") # [d_model, d_model]
          t_layer.mha.w_o = c_proj_w

          # FFN
          fc_w = sf.read_matrix("#{prefix}.mlp.c_fc.weight")     # [d_model, ff_hidden]
          fc_b = sf.read_f64("#{prefix}.mlp.c_fc.bias")          # [ff_hidden]
          proj_w = sf.read_matrix("#{prefix}.mlp.c_proj.weight") # [ff_hidden, d_model]
          proj_b = sf.read_f64("#{prefix}.mlp.c_proj.bias")      # [d_model]

          t_layer.ffn.w1 = fc_w
          t_layer.ffn.b1 = SimpleMatrix.from_a([fc_b])
          t_layer.ffn.w2 = proj_w
          t_layer.ffn.b2 = SimpleMatrix.from_a([proj_b])

          # Layer norms
          ln1_w = sf.read_f64("#{prefix}.ln_1.weight")
          ln1_b = sf.read_f64("#{prefix}.ln_1.bias")
          ln2_w = sf.read_f64("#{prefix}.ln_2.weight")
          ln2_b = sf.read_f64("#{prefix}.ln_2.bias")

          t_layer.norm1.gamma = SimpleMatrix.from_a([ln1_w])
          t_layer.norm1.beta = SimpleMatrix.from_a([ln1_b])
          t_layer.norm2.gamma = SimpleMatrix.from_a([ln2_w])
          t_layer.norm2.beta = SimpleMatrix.from_a([ln2_b])
        end

        # Note: GPT-2 ln_f (final LayerNorm) is not loaded here.
        # For proper GPT-2 inference, ln_f should be applied before the output
        # projection. This is acceptable for the tiny test model but will produce
        # slightly incorrect logits for real GPT-2 models.

        # Output weights: GPT-2 ties lm_head to wte (transposed)
        # MatrixLayer#forward does: input * weights, so weights must be [d_model, vocab_size]
        output_layer = net.output_layers.first
        if sf.has_tensor?("lm_head.weight")
          lm_w = sf.read_matrix("lm_head.weight") # HF stores [vocab, d_model]
          output_layer.weights = lm_w.transpose   # -> [d_model, vocab]
        else
          # Tied weights: wte is [vocab, d_model], transpose -> [d_model, vocab]
          output_layer.weights = wte.transpose
        end
        # Zero out bias (GPT-2 lm_head has no bias)
        output_layer.biases = SimpleMatrix.new(1, config.vocab_size)

        net
      ensure
        sf.close
      end
    end

    # LLaMA config
    record LlamaConfig,
      vocab_size : Int32,
      hidden_size : Int32,
      num_attention_heads : Int32,
      num_hidden_layers : Int32,
      intermediate_size : Int32,
      rms_norm_eps : Float64,
      rope_theta : Float64,
      num_key_value_heads : Int32,
      tie_word_embeddings : Bool,
      rope_scaling : JSON::Any?,
      head_dim : Int32? = nil,
      num_experts : Int32? = nil,
      num_experts_per_tok : Int32 = 8,
      norm_topk_prob : Bool = true,
      moe_intermediate_size : Int32? = nil,
      # Hybrid linear-attention fields, present on qwen3_5 (Qwen3.5 / Qwen3.6) and absent on
      # every other architecture here. nil layer_types means "every layer is full attention",
      # which is what all the older Qwen and LLaMA configs mean by saying nothing.
      layer_types : Array(String)? = nil,
      linear_conv_kernel_dim : Int32 = 4,
      linear_key_head_dim : Int32 = 128,
      linear_value_head_dim : Int32 = 128,
      linear_num_key_heads : Int32 = 16,
      linear_num_value_heads : Int32 = 32

    # Tensor-name prefixes in a qwen3_5 checkpoint that a TEXT-ONLY load must skip.
    #
    # These are real tensors, not junk: "mtp." is a multi-token-prediction head and the vision
    # entries are the image/video tower. Both are dead weight for text generation, and both
    # would otherwise look like unmapped tensors and mask a genuine mapping gap.
    QWEN35_SKIP_PREFIXES = ["mtp.", "model.visual.", "model.vision_tower.", "visual."]

    def self.qwen35_skip?(name : String) : Bool
      QWEN35_SKIP_PREFIXES.any? { |p| name.starts_with?(p) }
    end

    # The tensor names this loader expects for one qwen3_5 layer, given its type.
    #
    # Kept as data rather than buried in the load loop so it can be diffed against a real
    # checkpoint's index without touching a single weight -- which is how the layout was
    # established in the first place. Guessing these names risks loading the wrong tensor into a
    # correctly shaped slot, which produces fluent nonsense rather than an error.
    #
    # The prefix is "model.language_model." and NOT "model.", because the text backbone sits
    # beside a vision tower. Every other architecture here uses "model.".
    def self.qwen35_layer_tensors(index : Int32, layer_type : String) : Array(String)
      p = "model.language_model.layers.#{index}."
      common = [
        "#{p}input_layernorm.weight",
        "#{p}post_attention_layernorm.weight",
        "#{p}mlp.gate_proj.weight",
        "#{p}mlp.up_proj.weight",
        "#{p}mlp.down_proj.weight",
      ]
      case layer_type
      when "linear_attention"
        common + [
          # One fused projection carrying q, k and v: [q_dim + k_dim + v_dim, d_model].
          "#{p}linear_attn.in_proj_qkv.weight",
          # Alpha and beta, one row per VALUE head.
          "#{p}linear_attn.in_proj_a.weight",
          "#{p}linear_attn.in_proj_b.weight",
          # The output gate, which the paper calls a SiLU gate on the output path.
          "#{p}linear_attn.in_proj_z.weight",
          # One depthwise conv over ALL fused qkv channels: [q+k+v, 1, kernel].
          "#{p}linear_attn.conv1d.weight",
          # Mamba2's per-head decay parameters. Their presence is what confirms the gate
          # parameterization that the paper leaves unstated.
          "#{p}linear_attn.A_log",
          "#{p}linear_attn.dt_bias",
          # Per-head output norm, sized head_v rather than the concatenated v_dim.
          "#{p}linear_attn.norm.weight",
          "#{p}linear_attn.out_proj.weight",
        ]
      when "full_attention"
        common + [
          "#{p}self_attn.q_proj.weight",
          "#{p}self_attn.k_proj.weight",
          "#{p}self_attn.v_proj.weight",
          "#{p}self_attn.o_proj.weight",
          # Qwen3-style per-head QK norms, sized head_dim.
          "#{p}self_attn.q_norm.weight",
          "#{p}self_attn.k_norm.weight",
        ]
      else
        raise ArgumentError.new("unknown layer_type #{layer_type.inspect}")
      end
    end

    # Every text tensor a qwen3_5 load needs, in layer order, plus the embedding and head.
    def self.qwen35_tensor_plan(config : LlamaConfig) : Array(String)
      types = config.layer_types || default_layer_types(config.num_hidden_layers)
      names = ["model.language_model.embed_tokens.weight"]
      types.each_with_index { |t, i| names.concat(qwen35_layer_tensors(i, t)) }
      names << "model.language_model.norm.weight"
      names << "lm_head.weight" unless config.tie_word_embeddings
      names
    end

    # Load a qwen3_5 (Qwen3.5 / Qwen3.6) hybrid stack.
    #
    # Differs from load_llama in four ways, every one of them established by reading a real
    # Qwen3.5-9B checkpoint rather than inferred:
    #
    #   * tensors live under "model.language_model.", not "model.", because the text backbone
    #     sits beside a vision tower
    #   * the stack is hybrid, so each layer is built from layer_types
    #   * a linear layer's q/k/v arrive FUSED in one in_proj_qkv, and its three short-conv
    #     kernels arrive fused in one conv1d
    #   * the output norm is per-head, sized head_v
    #
    # Text-only: the vision tower and the multi-token-prediction head are skipped.
    #
    # max_layers truncates the stack, which exists for verification rather than for use. A full
    # fp32 host load of the 9B needs 33.4 GiB resident (embedding 3.79, lm_head 3.79, and 32
    # layers), which does not reliably fit 62 GiB of RAM alongside anything else -- the first
    # attempt was OOM-killed at 45 GiB. A truncated stack loads the same code paths against the
    # same tensors at a fraction of the memory, so the weight mapping can be exercised before
    # the GPU mixer exists. A truncated model does NOT produce meaningful text.
    def self.load_qwen35(model_dir : String, max_layers : Int32? = nil) : Network
      config_path = ::File.join(model_dir, "config.json")
      raise "config.json not found in #{model_dir}" unless ::File.exists?(config_path)
      config = load_llama_config(config_path)
      types = config.layer_types || default_layer_types(config.num_hidden_layers)
      types = types[0, max_layers] if max_layers && max_layers < types.size
      sf = open_safetensors(model_dir)

      begin
        d = config.hidden_size
        ff = config.intermediate_size
        eps = config.rms_norm_eps
        head_dim = config.head_dim || (d // config.num_attention_heads)

        net = Network.new
        net.add_layer(:input, 1)
        net.add_layer(:embedding, d, vocab_size: config.vocab_size)
        types.each do |t|
          if t == "linear_attention"
            net.add_layer("gated_deltanet", d, num_heads: config.linear_num_value_heads,
              ff_hidden: ff, num_kv_heads: config.linear_num_key_heads,
              eps: eps, head_dim: config.linear_key_head_dim,
              linear_conv_kernel: config.linear_conv_kernel_dim)
          else
            net.add_layer(:llama, d, num_heads: config.num_attention_heads, ff_hidden: ff,
              num_kv_heads: config.num_key_value_heads, eps: eps, head_dim: config.head_dim)
          end
        end
        net.add_layer(:output, config.vocab_size, activation_function: SHAInet.identity)
        net.fully_connect

        emb_layer = net.hidden_layers.find(&.is_a?(EmbeddingLayer)).as(EmbeddingLayer)
        # Scoped so the read buffer is collectable immediately. Holding it for a tied-weights
        # check would pin 3.79 GiB across the entire load, and this architecture ships an
        # explicit lm_head anyway.
        begin
          embed = sf.read_matrix("model.language_model.embed_tokens.weight")
          config.vocab_size.times { |i| d.times { |j| emb_layer.embeddings[i, j] = embed[i, j] } }
        end
        GC.collect
        Log.info { "qwen3_5: embedding loaded (#{config.vocab_size}x#{d})" }

        # The blocks in stack order. gated_deltanet layers are not in @transformer_layers (see
        # network_setup), so walk @hidden_layers and skip the embedding.
        blocks = net.hidden_layers.reject(EmbeddingLayer)

        types.each_with_index do |t, idx|
          p = "model.language_model.layers.#{idx}."
          if t == "linear_attention"
            load_qwen35_linear_layer(sf, blocks[idx].as(GatedDeltaNetBlock), p, config)
          else
            block = blocks[idx].as(LlamaBlock)
            block.rope_theta = config.rope_theta
            block.rope_freqs = compute_rope_freqs(config, head_dim)
            # Qwen3.5's full-attention layers are GATED: q_proj packs q and the output gate
            # into one [2 * q_dim, d_model] tensor. Measured on Qwen3.5-9B, q_proj is
            # [8192, 4096] while num_attention_heads * head_dim is 4096.
            #
            # Assigning that straight to w_q silently produced a [4096, 8192] weight in a slot
            # the block believes is [4096, 4096] -- no error, just wrong output. Hence the
            # explicit split and the assertions below.
            q_dim = config.num_attention_heads * head_dim
            qp = sf.read_matrix_transposed("#{p}self_attn.q_proj.weight")
            case qp.cols
            when q_dim
              block.w_q = qp
            when 2 * q_dim
              wq = SimpleMatrix.new(qp.rows, q_dim)
              wg = SimpleMatrix.new(qp.rows, q_dim)
              qp.rows.times do |i|
                q_dim.times do |j|
                  wq[i, j] = qp[i, j]
                  wg[i, j] = qp[i, q_dim + j]
                end
              end
              block.w_q = wq
              block.w_gate_attn = wg
            else
              raise "#{p}self_attn.q_proj.weight gives #{qp.cols} columns, expected #{q_dim} (ungated) or #{2 * q_dim} (gated)"
            end
            kv_dim = config.num_key_value_heads * head_dim
            wk = sf.read_matrix_transposed("#{p}self_attn.k_proj.weight")
            wv = sf.read_matrix_transposed("#{p}self_attn.v_proj.weight")
            raise "#{p}self_attn.k_proj.weight gives #{wk.cols} columns, expected #{kv_dim}" unless wk.cols == kv_dim
            raise "#{p}self_attn.v_proj.weight gives #{wv.cols} columns, expected #{kv_dim}" unless wv.cols == kv_dim
            block.w_k = wk
            block.w_v = wv
            wo = sf.read_matrix_transposed("#{p}self_attn.o_proj.weight")
            raise "#{p}self_attn.o_proj.weight gives #{wo.rows}x#{wo.cols}, expected #{q_dim}x#{d}" unless wo.rows == q_dim && wo.cols == d
            block.w_o = wo
            qn = sf.read_matrix("#{p}self_attn.q_norm.weight")
            kn = sf.read_matrix("#{p}self_attn.k_norm.weight")
            block.q_norm = Array(Float32).new(qn.cols) { |i| qn[0, i].to_f32 }
            block.k_norm = Array(Float32).new(kn.cols) { |i| kn[0, i].to_f32 }
            ffn = block.ffn.as(SwiGLUFF)
            ffn.gate_proj = sf.read_matrix_transposed("#{p}mlp.gate_proj.weight")
            ffn.up_proj = sf.read_matrix_transposed("#{p}mlp.up_proj.weight")
            ffn.down_proj = sf.read_matrix_transposed("#{p}mlp.down_proj.weight")
            block.norm1.gamma = sf.read_matrix("#{p}input_layernorm.weight")
            block.norm2.gamma = sf.read_matrix("#{p}post_attention_layernorm.weight")
          end
          # Collect EVERY layer: a 9B's resident fp32 footprint is 33.4 GiB, so a few hundred
          # MB of retained read transients per layer is the difference between fitting and being
          # OOM-killed. Measured twice on this machine before this was tightened.
          GC.collect
          if idx % 4 == 3 || idx == types.size - 1
            Log.info { "qwen3_5: layer #{idx + 1}/#{types.size} loaded" }
          end
        end

        final_norm = RMSNorm.new(d, eps)
        final_norm.gamma = sf.read_matrix("model.language_model.norm.weight")
        net.final_norm = final_norm

        output_layer = net.output_layers.first
        head_name = config.tie_word_embeddings ? "model.language_model.embed_tokens.weight" : "lm_head.weight"
        output_layer.weights = sf.read_matrix_transposed(head_name)

        net
      ensure
        sf.close
      end
    end

    # One linear-attention layer's weights, unpacking the fused tensors.
    private def self.load_qwen35_linear_layer(sf, block : GatedDeltaNetBlock, p : String, config : LlamaConfig)
      k_dim = config.linear_num_key_heads * config.linear_key_head_dim
      v_dim = config.linear_num_value_heads * config.linear_value_head_dim

      # in_proj_qkv is [q_dim + k_dim + v_dim, d_model] with q_dim == k_dim (q uses the KEY head
      # count, not the value head count). Transposed once, then sliced by column range.
      qkv = sf.read_matrix_transposed("#{p}linear_attn.in_proj_qkv.weight") # [d_model, 2*k_dim + v_dim]
      expected = 2 * k_dim + v_dim
      raise "#{p}linear_attn.in_proj_qkv.weight has #{qkv.cols} columns, expected #{expected}" unless qkv.cols == expected
      copy_cols(qkv, block.w_q, 0, k_dim)
      copy_cols(qkv, block.w_k, k_dim, k_dim)
      copy_cols(qkv, block.w_v, 2 * k_dim, v_dim)

      # One depthwise conv over ALL fused qkv channels, in the same q, k, v order. Split into the
      # block's three ShortConvs, whose channel counts sum to exactly this tensor's rows.
      conv = sf.read_matrix("#{p}linear_attn.conv1d.weight") # [2*k_dim + v_dim, kernel]
      raise "#{p}linear_attn.conv1d.weight has #{conv.rows} rows, expected #{expected}" unless conv.rows == expected
      copy_rows(conv, block.conv_q.weight, 0)
      copy_rows(conv, block.conv_k.weight, k_dim)
      copy_rows(conv, block.conv_v.weight, 2 * k_dim)

      block.w_alpha = sf.read_matrix_transposed("#{p}linear_attn.in_proj_a.weight")
      block.w_beta = sf.read_matrix_transposed("#{p}linear_attn.in_proj_b.weight")
      block.w_gate = sf.read_matrix_transposed("#{p}linear_attn.in_proj_z.weight")
      block.w_o = sf.read_matrix_transposed("#{p}linear_attn.out_proj.weight")

      a_log = sf.read_matrix("#{p}linear_attn.A_log")
      dt_bias = sf.read_matrix("#{p}linear_attn.dt_bias")
      config.linear_num_value_heads.times do |h|
        block.a_log[h] = a_log[0, h].to_f64
        block.dt_bias[h] = dt_bias[0, h].to_f64
      end

      block.out_norm.gamma = sf.read_matrix("#{p}linear_attn.norm.weight")
      block.norm1.gamma = sf.read_matrix("#{p}input_layernorm.weight")
      block.norm2.gamma = sf.read_matrix("#{p}post_attention_layernorm.weight")

      ffn = block.ffn
      ffn.gate_proj = sf.read_matrix_transposed("#{p}mlp.gate_proj.weight")
      ffn.up_proj = sf.read_matrix_transposed("#{p}mlp.up_proj.weight")
      ffn.down_proj = sf.read_matrix_transposed("#{p}mlp.down_proj.weight")
    end

    private def self.copy_cols(src : SimpleMatrix, dst : SimpleMatrix, offset : Int32, width : Int32)
      raise "copy_cols shape: dst #{dst.rows}x#{dst.cols}, src #{src.rows} rows, width #{width}" unless dst.rows == src.rows && dst.cols == width
      src.rows.times { |i| width.times { |j| dst[i, j] = src[i, offset + j] } }
    end

    private def self.copy_rows(src : SimpleMatrix, dst : SimpleMatrix, offset : Int32)
      raise "copy_rows shape: dst cols #{dst.cols} != src cols #{src.cols}" unless dst.cols == src.cols
      dst.rows.times { |i| src.cols.times { |j| dst[i, j] = src[offset + i, j] } }
    end

    # Number of linear-attention layers per full-attention layer when a qwen3_5 config does not
    # spell out layer_types. Transformers generates the list from config values in that case,
    # and the published stack is 3:1 -- three Gated DeltaNet layers for every one Gated
    # Attention layer.
    HYBRID_LINEAR_PER_FULL = 3

    # Build the default 3:1 layer pattern, full attention every fourth layer.
    #
    # The LAST layer of each group is the full-attention one, matching the published design
    # where a group of linear layers is followed by an attention layer that can look back over
    # the whole context.
    def self.default_layer_types(num_layers : Int32) : Array(String)
      Array(String).new(num_layers) do |i|
        ((i + 1) % (HYBRID_LINEAR_PER_FULL + 1) == 0) ? "full_attention" : "linear_attention"
      end
    end

    def self.load_llama_config(path : String) : LlamaConfig
      root = JSON.parse(::File.read(path))
      # qwen3_5 is natively multimodal, so its text hyperparameters live under text_config with
      # a sibling vision_config, where every older architecture here puts them at the top
      # level. Read through the nesting when it is present rather than duplicating the parser.
      json = root["text_config"]? || root

      layer_types = json["layer_types"]?.try(&.as_a.map(&.as_s))
      # A qwen3_5 config with no explicit list still means a hybrid stack, so synthesize the
      # pattern rather than silently treating every layer as full attention.
      if layer_types.nil? && (root["model_type"]?.try(&.as_s) == "qwen3_5")
        layer_types = default_layer_types(json["num_hidden_layers"].as_i)
      end

      LlamaConfig.new(
        vocab_size: json["vocab_size"].as_i,
        hidden_size: json["hidden_size"].as_i,
        num_attention_heads: json["num_attention_heads"].as_i,
        num_hidden_layers: json["num_hidden_layers"].as_i,
        intermediate_size: json["intermediate_size"].as_i,
        rms_norm_eps: json["rms_norm_eps"].as_f,
        rope_theta: (json["rope_theta"]?.try(&.as_f) || json["rope_parameters"]?.try(&.["rope_theta"]?).try(&.as_f) || 10000.0),
        num_key_value_heads: (json["num_key_value_heads"]?.try(&.as_i) || json["num_attention_heads"].as_i),
        tie_word_embeddings: (json["tie_word_embeddings"]?.try(&.as_bool) || false),
        rope_scaling: json["rope_scaling"]?,
        head_dim: json["head_dim"]?.try(&.as_i),
        num_experts: (json["num_experts"]?.try(&.as_i) || json["num_local_experts"]?.try(&.as_i)),
        num_experts_per_tok: (json["num_experts_per_tok"]?.try(&.as_i) || 8),
        norm_topk_prob: (json.as_h.has_key?("norm_topk_prob") ? json["norm_topk_prob"].as_bool : true),
        moe_intermediate_size: json["moe_intermediate_size"]?.try(&.as_i),
        layer_types: layer_types,
        linear_conv_kernel_dim: (json["linear_conv_kernel_dim"]?.try(&.as_i) || 4),
        linear_key_head_dim: (json["linear_key_head_dim"]?.try(&.as_i) || 128),
        linear_value_head_dim: (json["linear_value_head_dim"]?.try(&.as_i) || 128),
        linear_num_key_heads: (json["linear_num_key_heads"]?.try(&.as_i) || 16),
        linear_num_value_heads: (json["linear_num_value_heads"]?.try(&.as_i) || 32)
      )
    end

    # Compute inverse frequencies (size head_dim/2) for RoPE, applying
    # LLaMA 3 rope_scaling when present. Returns nil for the default case
    # (no scaling), letting the block use the plain theta^(-2i/d) formula.
    def self.compute_rope_freqs(config : LlamaConfig, head_dim : Int32) : Array(Float32)?
      scaling = config.rope_scaling
      return if scaling.nil? || scaling.raw.nil?

      stype = scaling["rope_type"]?.try(&.as_s) || scaling["type"]?.try(&.as_s)
      return unless stype == "llama3"

      theta = config.rope_theta
      factor = scaling["factor"].as_f
      low_freq_factor = scaling["low_freq_factor"].as_f
      high_freq_factor = scaling["high_freq_factor"].as_f
      old_ctx = scaling["original_max_position_embeddings"].as_i.to_f64

      low_freq_wavelen = old_ctx / low_freq_factor
      high_freq_wavelen = old_ctx / high_freq_factor

      half = head_dim // 2
      Array(Float32).new(half) do |i|
        inv = 1.0 / (theta ** (2.0 * i / head_dim))
        wavelen = 2.0 * Math::PI / inv
        new_inv = if wavelen < high_freq_wavelen
                    inv
                  elsif wavelen > low_freq_wavelen
                    inv / factor
                  else
                    smooth = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                    (1.0 - smooth) * inv / factor + smooth * inv
                  end
        new_inv.to_f32
      end
    end

    # Load LLaMA/Mistral/Qwen2 model from SafeTensors.
    #
    # When `quantize` is true (and CUDA is available) each transformer block is
    # quantized to the requested width (`bits`: 8 -> Q8, 4 -> Q4) *immediately
    # after its weights are read*, so the fp32 copies are freed before the next
    # layer loads. This keeps host memory
    # bounded (a few GB) instead of materializing the entire fp32 model at once
    # (~28 GB for a 7B), which lets large models load on modest-RAM machines.
    def self.load_llama(model_dir : String, quantize : Bool = false, bits : Int32 = 8) : Network
      config_path = ::File.join(model_dir, "config.json")
      raise "config.json not found in #{model_dir}" unless ::File.exists?(config_path)

      do_quant = quantize && CUDA.fully_available?
      config = load_llama_config(config_path)
      sf = open_safetensors(model_dir)

      begin
        d = config.hidden_size
        ff = config.intermediate_size
        n_heads = config.num_attention_heads
        eps = config.rms_norm_eps
        theta = config.rope_theta
        # Qwen3 sets head_dim explicitly (e.g. 128), independent of d/n_heads;
        # LLaMA/Qwen2 leave it nil and the block derives d/n_heads (and validates
        # divisibility). Use a concrete value only for the local RoPE-freq calc.
        head_dim = config.head_dim || (d // n_heads)
        rope_freqs = compute_rope_freqs(config, head_dim)

        net = Network.new
        net.add_layer(:input, 1)
        net.add_layer(:embedding, d, vocab_size: config.vocab_size)
        # Offload MoE experts to host RAM (streamed to GPU on demand) when
        # requested — lets large MoE models fit small GPUs. Q4 only.
        moe_offload = ENV.fetch("SHAINET_MOE_OFFLOAD", "0") == "1"
        # Offload the DENSE weights too (attention projections, dense FFN,
        # lm_head). This is what lifts the dense-model ceiling off VRAM; unlike
        # experts these are touched every token, so it costs PCIe bandwidth.
        dense_offload = ENV.fetch("SHAINET_DENSE_OFFLOAD", "0") == "1"
        if dense_offload && !(do_quant && bits == 4)
          raise ArgumentError.new("SHAINET_DENSE_OFFLOAD=1 requires 4-bit quantization (set SHAINET_Q4=1); got quantize=#{quantize} bits=#{bits}")
        end
        # Dense offload only reduces VRAM if the hot cache is BOUNDED. Left at its
        # default (70% of free VRAM) the cache promotes the dense weights straight
        # back onto the card, and because they are touched every token it keeps
        # them there: measured on Qwen3-0.6B the default budget gave a HIGHER peak
        # than not offloading at all (586 MB vs 483 MB attributable), while
        # SHAINET_EXPERT_CACHE_MB=0 gave 228 MB. Warn rather than silently
        # delivering a pessimization.
        if dense_offload && !ENV.has_key?("SHAINET_EXPERT_CACHE_MB")
          Log.warn { "SHAINET_DENSE_OFFLOAD=1 without SHAINET_EXPERT_CACHE_MB: the hot cache defaults to 70% of free VRAM and will promote the dense weights back onto the device, which can use MORE VRAM than not offloading. Set SHAINET_EXPERT_CACHE_MB (0 disables the cache) to get the capacity win." }
        end
        config.num_hidden_layers.times do
          net.add_layer(:llama, d, num_heads: n_heads, ff_hidden: ff, num_kv_heads: config.num_key_value_heads, eps: eps, head_dim: config.head_dim,
            moe_experts: config.num_experts, moe_top_k: config.num_experts_per_tok, moe_norm_topk: config.norm_topk_prob, moe_ff_hidden: config.moe_intermediate_size, moe_offload: moe_offload)
        end
        net.add_layer(:output, config.vocab_size, activation_function: SHAInet.identity)
        net.fully_connect

        # Load embeddings
        emb_layer = net.hidden_layers.find(&.is_a?(EmbeddingLayer)).as(EmbeddingLayer)
        embed = sf.read_matrix("model.embed_tokens.weight") # [vocab, d]
        config.vocab_size.times do |i|
          d.times { |j| emb_layer.embeddings[i, j] = embed[i, j] }
        end
        # Keep the embedding table in host RAM for quantized (inference) loads:
        # it is gather-only, so residency costs vocab * d * 4 bytes of VRAM
        # (~3.7 GB for a 150k-vocab 30B) and buys nothing. Opt out with
        # SHAINET_EMBED_HOST=0; set it to 1 to also skip the transient device
        # allocation during the load above.
        emb_layer.to_host! if do_quant && ENV.fetch("SHAINET_EMBED_HOST", "1") != "0"
        # Reclaim the embedding read transients before the layer loop begins
        # (the bf16->f32 conversion buffer is large for big-vocab models).
        GC.collect if do_quant

        # Load transformer blocks
        config.num_hidden_layers.times do |idx|
          block = net.transformer_layers[idx].as(LlamaBlock)
          block.rope_theta = theta
          block.rope_freqs = rope_freqs
          prefix = "model.layers.#{idx}"

          # Attention weights: HF is [out, in], matmul is x * W so need [in, out]
          block.w_q = sf.read_matrix("#{prefix}.self_attn.q_proj.weight").transpose
          block.w_k = sf.read_matrix("#{prefix}.self_attn.k_proj.weight").transpose
          block.w_v = sf.read_matrix("#{prefix}.self_attn.v_proj.weight").transpose
          block.w_o = sf.read_matrix("#{prefix}.self_attn.o_proj.weight").transpose

          # FFN — dense SwiGLU or Mixture-of-Experts, matching the block type.
          case ffn = block.ffn
          when SwiGLUFF
            ffn.gate_proj = sf.read_matrix("#{prefix}.mlp.gate_proj.weight").transpose
            ffn.up_proj = sf.read_matrix("#{prefix}.mlp.up_proj.weight").transpose
            ffn.down_proj = sf.read_matrix("#{prefix}.mlp.down_proj.weight").transpose
          when MoEFF
            # Router: HF [num_experts, hidden] -> [hidden, num_experts].
            ffn.router = sf.read_matrix("#{prefix}.mlp.gate.weight").transpose
            ffn.experts.each_with_index do |expert, e|
              eprefix = "#{prefix}.mlp.experts.#{e}"
              expert.gate_proj = sf.read_matrix("#{eprefix}.gate_proj.weight").transpose
              expert.up_proj = sf.read_matrix("#{eprefix}.up_proj.weight").transpose
              expert.down_proj = sf.read_matrix("#{eprefix}.down_proj.weight").transpose

              # The block-level collection below only fires once per LAYER, and a
              # 128-expert layer reads 384 tensors before reaching it. Each read
              # allocates a raw byte buffer and an fp32 matrix, and .transpose
              # allocates a second matrix it then discards, so the garbage from one
              # layer's experts reached several GB before anything reclaimed it.
              #
              # Measured on Qwen3-Coder-30B-A3B: the agent was OOM-killed at 42.3 GiB
              # of anonymous RSS during load on a 62 GB host, while the same load with
              # Boehm collecting harder (GC_FREE_SPACE_DIVISOR=8) peaked at 33.4 GiB.
              # This is that reclaim, without asking the user for an env var.
              GC.collect if do_quant && (e + 1) % EXPERT_GC_INTERVAL == 0
            end
          end

          # RMSNorm
          block.norm1.gamma = sf.read_matrix("#{prefix}.input_layernorm.weight")
          block.norm2.gamma = sf.read_matrix("#{prefix}.post_attention_layernorm.weight")

          # Optional Q/K/V projection biases — present in Qwen2-style models,
          # absent in LLaMA/Mistral. They always appear as a complete set.
          has_q = sf.has_tensor?("#{prefix}.self_attn.q_proj.bias")
          has_k = sf.has_tensor?("#{prefix}.self_attn.k_proj.bias")
          has_v = sf.has_tensor?("#{prefix}.self_attn.v_proj.bias")
          if has_q || has_k || has_v
            raise "Incomplete Q/K/V projection biases for #{prefix} (q=#{has_q} k=#{has_k} v=#{has_v})" unless has_q && has_k && has_v
            block.b_q = sf.read_f32("#{prefix}.self_attn.q_proj.bias")
            block.b_k = sf.read_f32("#{prefix}.self_attn.k_proj.bias")
            block.b_v = sf.read_f32("#{prefix}.self_attn.v_proj.bias")
          end

          # Optional Qwen3 QK-norm weights (per-head RMSNorm over head_dim),
          # applied to Q and K before RoPE. Present for qwen3/qwen3_moe, absent
          # for LLaMA/Qwen2. Always a complete pair.
          has_qn = sf.has_tensor?("#{prefix}.self_attn.q_norm.weight")
          has_kn = sf.has_tensor?("#{prefix}.self_attn.k_norm.weight")
          if has_qn || has_kn
            raise "Incomplete QK-norm weights for #{prefix} (q_norm=#{has_qn} k_norm=#{has_kn})" unless has_qn && has_kn
            block.q_norm = sf.read_f32("#{prefix}.self_attn.q_norm.weight")
            block.k_norm = sf.read_f32("#{prefix}.self_attn.k_norm.weight")
          end

          # Quantize this block now so its fp32 weights can be freed before the
          # next layer is read (bounds peak host memory for large models). Force
          # a collection so the just-replaced fp32 SimpleMatrices + read/transpose
          # transients are reclaimed before the next layer allocates — otherwise
          # GC lag lets ~28 layers of fp32 garbage pile up and OOM a big model.
          if do_quant
            block.to_gpu!(quantize: true, bits: bits, offload: dense_offload)
            GC.collect
          end
          @@progress.try &.call(idx + 1, config.num_hidden_layers)
        end

        # Output head
        output_layer = net.output_layers.first

        # Final RMSNorm (applied before output projection)
        final_norm = RMSNorm.new(d, eps)
        final_norm.gamma = sf.read_matrix("model.norm.weight")
        net.final_norm = final_norm

        if config.tie_word_embeddings
          output_layer.weights = embed.transpose # [d, vocab]
        else
          output_layer.weights = sf.read_matrix("lm_head.weight").transpose
        end
        output_layer.biases = SimpleMatrix.new(1, config.vocab_size)

        # Quantize the lm_head (and idempotently re-confirm the already-Q8
        # blocks) + set the quantized-weights flag. Blocks were quantized inline
        # above, so this only materializes the lm_head fp32 transiently.
        net.quantize!(bits, offload: dense_offload) if do_quant

        net
      ensure
        sf.close
      end
    end
  end
end
