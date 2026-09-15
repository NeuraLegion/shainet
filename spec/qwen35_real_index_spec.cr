require "./spec_helper"
require "json"

# Diffs the loader's expected tensor names against a REAL Qwen3.5 checkpoint index.
#
# This is the check that closed the last gap. The mapping cannot be guessed safely: loading the
# wrong tensor into a correctly shaped slot produces fluent nonsense rather than an error, and
# three of my assumptions were wrong until this was run --
#
#   * the prefix is "model.language_model." not "model.", because the text backbone sits beside
#     a vision tower
#   * projections are FUSED (one in_proj_qkv) rather than separate q/k/v
#   * the output norm is per-head [head_v], not over the concatenated v_dim
#
# It needs only model.safetensors.index.json, a few hundred KB, not the weights. When that file
# is absent the examples are pending rather than failing, so the suite still runs on a machine
# without the checkpoint.
QWEN35_DIR   = "/home/unshadow/models/Qwen3.5-9B"
QWEN35_INDEX = File.join(QWEN35_DIR, "model.safetensors.index.json")

def real_tensor_names : Array(String)
  JSON.parse(File.read(QWEN35_INDEX))["weight_map"].as_h.keys
end

describe "qwen3_5 tensor plan against a real checkpoint" do
  it "expects only tensors the checkpoint actually contains" do
    pending! "#{QWEN35_INDEX} not present" unless File.exists?(QWEN35_INDEX)

    config = SHAInet::HFLoader.load_llama_config(File.join(QWEN35_DIR, "config.json"))
    plan = SHAInet::HFLoader.qwen35_tensor_plan(config)
    have = real_tensor_names.to_set

    missing = plan.reject { |n| have.includes?(n) }
    # A name in the plan that the checkpoint lacks would raise mid-load on a real run.
    missing.should be_empty
  end

  it "covers every text tensor the checkpoint provides" do
    pending! "#{QWEN35_INDEX} not present" unless File.exists?(QWEN35_INDEX)

    config = SHAInet::HFLoader.load_llama_config(File.join(QWEN35_DIR, "config.json"))
    plan = SHAInet::HFLoader.qwen35_tensor_plan(config).to_set

    # Everything not skipped and not planned is a genuine gap: a weight the model needs that the
    # loader would leave at its initial value, which is silent wrongness rather than a crash.
    unmapped = real_tensor_names.reject do |n|
      SHAInet::HFLoader.qwen35_skip?(n) || plan.includes?(n)
    end
    unmapped.should be_empty
  end

  it "classifies the multi-token-prediction head and vision tower as skipped" do
    pending! "#{QWEN35_INDEX} not present" unless File.exists?(QWEN35_INDEX)

    names = real_tensor_names
    mtp = names.count(&.starts_with?("mtp."))
    # The checkpoint really does carry an mtp head; if this ever hits zero the skip list is
    # matching nothing and the coverage example above has gone vacuous.
    mtp.should be > 0
    names.select(&.starts_with?("mtp.")).all? { |n| SHAInet::HFLoader.qwen35_skip?(n) }.should be_true
  end

  it "reads the real config's hybrid layout" do
    pending! "#{File.join(QWEN35_DIR, "config.json")} not present" unless File.exists?(File.join(QWEN35_DIR, "config.json"))

    config = SHAInet::HFLoader.load_llama_config(File.join(QWEN35_DIR, "config.json"))
    types = config.layer_types.not_nil!

    # The published 3:1 stack, from the actual file rather than a synthetic fixture.
    config.num_hidden_layers.should eq(32)
    types.count("linear_attention").should eq(24)
    types.count("full_attention").should eq(8)
    config.linear_num_value_heads.should eq(32)
    config.linear_num_key_heads.should eq(16)
    config.linear_key_head_dim.should eq(128)
    config.linear_conv_kernel_dim.should eq(4)
  end

  it "generates exactly the checkpoint's own layer pattern when the list is dropped" do
    pending! "#{File.join(QWEN35_DIR, "config.json")} not present" unless File.exists?(File.join(QWEN35_DIR, "config.json"))

    config = SHAInet::HFLoader.load_llama_config(File.join(QWEN35_DIR, "config.json"))
    real = config.layer_types.not_nil!
    synthesized = SHAInet::HFLoader.default_layer_types(config.num_hidden_layers)

    # Validates the fallback against ground truth: a config that omits layer_types must still
    # produce the right stack.
    synthesized.should eq(real)
  end
end
