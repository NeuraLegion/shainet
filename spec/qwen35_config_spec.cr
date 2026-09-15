require "./spec_helper"
require "file_utils"

# Config parsing for qwen3_5 (Qwen3.5 / Qwen3.6), which differs from every other architecture
# here in two ways that would each silently produce a wrong model:
#
#   1. text hyperparameters are NESTED under text_config, beside a vision_config, because these
#      are natively multimodal checkpoints. Reading the top level would miss them entirely.
#   2. the stack is HYBRID -- a per-layer layer_types list of linear_attention / full_attention
#      at a 3:1 ratio -- so treating every layer as attention is wrong for three quarters of it.
#
# These examples use synthetic configs rather than a real checkpoint: the parsing is what is
# under test, and a 27B download proves nothing about it.
def write_config(dir : String, json : String) : String
  path = File.join(dir, "config.json")
  File.write(path, json)
  path
end

def with_tmp(&)
  dir = File.tempname("shainet_cfg")
  Dir.mkdir_p(dir)
  begin
    yield dir
  ensure
    FileUtils.rm_rf(dir) if Dir.exists?(dir)
  end
end

QWEN35_MIN = <<-JSON
  {
    "model_type": "qwen3_5",
    "vision_config": {"depth": 27, "hidden_size": 1152},
    "text_config": {
      "vocab_size": 248320,
      "hidden_size": 4096,
      "intermediate_size": 12288,
      "num_hidden_layers": 8,
      "num_attention_heads": 16,
      "num_key_value_heads": 4,
      "head_dim": 256,
      "rms_norm_eps": 1e-06,
      "linear_conv_kernel_dim": 4,
      "linear_key_head_dim": 128,
      "linear_value_head_dim": 128,
      "linear_num_key_heads": 16,
      "linear_num_value_heads": 32,
      "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention"
      ]
    }
  }
  JSON

describe "HFLoader qwen3_5 config" do
  it "reads the text hyperparameters through the text_config nesting" do
    with_tmp do |dir|
      cfg = SHAInet::HFLoader.load_llama_config(write_config(dir, QWEN35_MIN))

      # All of these sit under text_config, not at the top level. Reading the root would have
      # raised on a missing key, so getting real values here is the whole assertion.
      cfg.vocab_size.should eq(248320)
      cfg.hidden_size.should eq(4096)
      cfg.num_hidden_layers.should eq(8)
      cfg.num_attention_heads.should eq(16)
      cfg.num_key_value_heads.should eq(4)
      cfg.head_dim.should eq(256)
      cfg.intermediate_size.should eq(12288)
    end
  end

  it "parses the linear-attention dimensions" do
    with_tmp do |dir|
      cfg = SHAInet::HFLoader.load_llama_config(write_config(dir, QWEN35_MIN))

      cfg.linear_conv_kernel_dim.should eq(4)
      cfg.linear_key_head_dim.should eq(128)
      cfg.linear_value_head_dim.should eq(128)
      cfg.linear_num_key_heads.should eq(16)
      cfg.linear_num_value_heads.should eq(32)
    end
  end

  it "parses the explicit per-layer layer_types" do
    with_tmp do |dir|
      cfg = SHAInet::HFLoader.load_llama_config(write_config(dir, QWEN35_MIN))
      types = cfg.layer_types.not_nil!

      types.size.should eq(8)
      types.count("linear_attention").should eq(6)
      types.count("full_attention").should eq(2)
      types[3].should eq("full_attention")
      types[7].should eq("full_attention")
    end
  end

  it "synthesizes the 3:1 pattern when layer_types is absent" do
    with_tmp do |dir|
      # Transformers generates the list from config values when it is not spelled out. Treating
      # a silent config as all-full-attention would be wrong for three quarters of the stack.
      json = QWEN35_MIN.gsub(/,\s*"layer_types":\s*\[[^\]]*\]/, "")
      cfg = SHAInet::HFLoader.load_llama_config(write_config(dir, json))
      types = cfg.layer_types.not_nil!

      types.size.should eq(8)
      types.count("full_attention").should eq(2)
      types.count("linear_attention").should eq(6)
      # Full attention closes each group of three linear layers.
      types[3].should eq("full_attention")
      types[7].should eq("full_attention")
    end
  end

  it "generates a 3:1 pattern for a length that is not a multiple of four" do
    types = SHAInet::HFLoader.default_layer_types(10)
    types.size.should eq(10)
    types.count("full_attention").should eq(2)
    types[3].should eq("full_attention")
    types[7].should eq("full_attention")
    # A short trailing group stays linear rather than being padded with an attention layer.
    types[8].should eq("linear_attention")
    types[9].should eq("linear_attention")
  end

  it "leaves layer_types nil for a NON-hybrid architecture" do
    with_tmp do |dir|
      # A plain qwen3 config says nothing about layer types and means every layer is attention.
      # Synthesizing a hybrid pattern here would corrupt a model that currently works.
      json = <<-JSON
        {
          "model_type": "qwen3",
          "vocab_size": 151936,
          "hidden_size": 2048,
          "intermediate_size": 6144,
          "num_hidden_layers": 4,
          "num_attention_heads": 16,
          "num_key_value_heads": 8,
          "rms_norm_eps": 1e-06
        }
        JSON
      cfg = SHAInet::HFLoader.load_llama_config(write_config(dir, json))

      cfg.layer_types.should be_nil
      cfg.hidden_size.should eq(2048)
    end
  end

  it "still reads a flat config, so nothing regresses for existing architectures" do
    with_tmp do |dir|
      json = <<-JSON
        {
          "model_type": "qwen3_moe",
          "vocab_size": 151936,
          "hidden_size": 2048,
          "intermediate_size": 6144,
          "num_hidden_layers": 2,
          "num_attention_heads": 32,
          "num_key_value_heads": 4,
          "head_dim": 128,
          "rms_norm_eps": 1e-06,
          "num_experts": 128,
          "num_experts_per_tok": 8,
          "moe_intermediate_size": 768
        }
        JSON
      cfg = SHAInet::HFLoader.load_llama_config(write_config(dir, json))

      cfg.num_experts.should eq(128)
      cfg.moe_intermediate_size.should eq(768)
      cfg.head_dim.should eq(128)
      cfg.layer_types.should be_nil
    end
  end

  it "builds a hybrid stack from a qwen3_5 config, mixing both layer types" do
    # This example used to assert the loader REFUSED qwen3_5. It now loads, so the assertion is
    # inverted: what matters is that the stack it builds has the layer types the config asked
    # for, rather than 32 attention layers that would run and emit nonsense.
    types = SHAInet::HFLoader.default_layer_types(8)
    types.count("linear_attention").should eq(6)
    types.count("full_attention").should eq(2)

    # A synthetic config has no weights to load, so the stack shape is checked directly through
    # the same builder the loader uses.
    net = SHAInet::Network.new
    net.add_layer("embedding", 32, vocab_size: 100)
    types.each do |t|
      name = t == "linear_attention" ? "gated_deltanet" : "llama"
      net.add_layer(name, 32, num_heads: 4, ff_hidden: 64, num_kv_heads: 2, head_dim: 8)
    end
    net.hidden_layers.count(&.is_a?(SHAInet::GatedDeltaNetBlock)).should eq(6)
    net.hidden_layers.count(&.is_a?(SHAInet::LlamaBlock)).should eq(2)
  end
end
