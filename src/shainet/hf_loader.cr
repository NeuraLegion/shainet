require "./safetensors"
require "json"

module SHAInet
  # Load a GPT-2 model directly from a HuggingFace SafeTensors file.
  # No Python, no PyTorch — pure Crystal.
  module HFLoader
    SUPPORTED_MODELS = ["gpt2"]

    # Generic entry point — reads config.json and dispatches to the right loader.
    def self.load(model_dir : String) : Network
      config_path = ::File.join(model_dir, "config.json")
      raise "config.json not found in #{model_dir}" unless ::File.exists?(config_path)

      json = JSON.parse(::File.read(config_path))
      model_type = json["model_type"]?.try(&.as_s) || raise "No model_type in config.json"

      case model_type
      when "gpt2"
        load_gpt2(model_dir)
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
          t_layer = net.transformer_layers[idx]
          prefix = "transformer.h.#{idx}"

          # Set positional encoding on the block
          t_layer.positional_encoding = wpe

          # Attention: GPT-2 stores QKV combined as c_attn [d_model, 3*d_model]
          # Need to split into Q, K, V weight matrices
          c_attn_w = sf.read_matrix("#{prefix}.attn.c_attn.weight") # [d_model, 3*d_model]
          c_attn_b = sf.read_f64("#{prefix}.attn.c_attn.bias")      # [3*d_model]

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

        # Final layer norm (applied before output projection in GPT-2)
        # Store it on the last transformer layer's norm2 or handle during forward pass
        # For now, we'll need to handle ln_f separately
        # TODO: The network architecture needs a final LayerNorm before the output head

        # Output weights: GPT-2 ties lm_head to wte (transposed)
        # MatrixLayer#forward does: input * weights, so weights must be [d_model, vocab_size]
        output_layer = net.output_layers.first
        if sf.has_tensor?("lm_head.weight")
          lm_w = sf.read_matrix("lm_head.weight") # HF stores [vocab, d_model]
          output_layer.weights = lm_w.transpose    # -> [d_model, vocab]
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
  end
end
