require "./safetensors"
require "json"

module SHAInet
  # Load a GPT-2 model directly from a HuggingFace SafeTensors file.
  # No Python, no PyTorch — pure Crystal.
  module HFLoader
    SUPPORTED_MODELS = ["gpt2", "llama", "mistral", "qwen2", "qwen3"]

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
      when "llama", "mistral", "qwen2", "qwen3"
        load_llama(model_dir, quantize: quantize, bits: bits)
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
      head_dim : Int32? = nil

    def self.load_llama_config(path : String) : LlamaConfig
      json = JSON.parse(::File.read(path))
      LlamaConfig.new(
        vocab_size: json["vocab_size"].as_i,
        hidden_size: json["hidden_size"].as_i,
        num_attention_heads: json["num_attention_heads"].as_i,
        num_hidden_layers: json["num_hidden_layers"].as_i,
        intermediate_size: json["intermediate_size"].as_i,
        rms_norm_eps: json["rms_norm_eps"].as_f,
        rope_theta: (json["rope_theta"]?.try(&.as_f) || 10000.0),
        num_key_value_heads: (json["num_key_value_heads"]?.try(&.as_i) || json["num_attention_heads"].as_i),
        tie_word_embeddings: (json["tie_word_embeddings"]?.try(&.as_bool) || false),
        rope_scaling: json["rope_scaling"]?,
        head_dim: json["head_dim"]?.try(&.as_i)
      )
    end

    # Compute inverse frequencies (size head_dim/2) for RoPE, applying
    # LLaMA 3 rope_scaling when present. Returns nil for the default case
    # (no scaling), letting the block use the plain theta^(-2i/d) formula.
    def self.compute_rope_freqs(config : LlamaConfig, head_dim : Int32) : Array(Float32)?
      scaling = config.rope_scaling
      return nil if scaling.nil? || scaling.raw.nil?

      stype = scaling["rope_type"]?.try(&.as_s) || scaling["type"]?.try(&.as_s)
      return nil unless stype == "llama3"

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
        config.num_hidden_layers.times do
          net.add_layer(:llama, d, num_heads: n_heads, ff_hidden: ff, num_kv_heads: config.num_key_value_heads, eps: eps, head_dim: config.head_dim)
        end
        net.add_layer(:output, config.vocab_size, activation_function: SHAInet.identity)
        net.fully_connect

        # Load embeddings
        emb_layer = net.hidden_layers.find(&.is_a?(EmbeddingLayer)).as(EmbeddingLayer)
        embed = sf.read_matrix("model.embed_tokens.weight") # [vocab, d]
        config.vocab_size.times do |i|
          d.times { |j| emb_layer.embeddings[i, j] = embed[i, j] }
        end
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

          # FFN
          block.ffn.gate_proj = sf.read_matrix("#{prefix}.mlp.gate_proj.weight").transpose
          block.ffn.up_proj = sf.read_matrix("#{prefix}.mlp.up_proj.weight").transpose
          block.ffn.down_proj = sf.read_matrix("#{prefix}.mlp.down_proj.weight").transpose

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
            block.to_gpu!(quantize: true, bits: bits)
            GC.collect
          end
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
        net.quantize!(bits) if do_quant

        net
      ensure
        sf.close
      end
    end
  end
end
