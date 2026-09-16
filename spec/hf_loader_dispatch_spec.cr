require "./spec_helper"
require "file_utils"

# HFLoader.load dispatches on model_type. The qwen3_5 arm dropped `quantize` and `bits`, so a caller
# asking for Q4 got a full fp32 load: 33.9 GiB resident for a 9B, which the OOM killer ends on a
# 62 GB machine. examples/agent.cr goes through exactly this path, so the omission made the agent
# unusable with the model while every other entry point worked.
#
# Asserted by parsing the method's own signature rather than by loading a 19 GB checkpoint, so it
# runs in CI: what needs pinning is that the arguments are forwarded at all.

describe "HFLoader.load dispatch" do
  it "lists qwen3_5 as supported" do
    SHAInet::HFLoader::SUPPORTED_MODELS.should contain("qwen3_5")
  end

  it "forwards quantize and bits to the qwen3_5 loader" do
    src = ::File.read(::File.join(__DIR__, "..", "src", "shainet", "hf_loader.cr"))
    arm = src[src.index!(%(when "qwen3_5")), 400]
    call = arm[arm.index!("load_qwen35("), 120]
    call.should contain("quantize: quantize")
    call.should contain("bits: bits")
  end

  it "names qwen3_5 in the refusal for an unknown model_type" do
    # The refusal lists SUPPORTED_MODELS, so a user with a qwen3_5 checkpoint and a typo'd config
    # sees that the architecture is supported rather than concluding it is not.
    dir = ::File.tempname("hfdisp")
    Dir.mkdir_p(dir)
    begin
      ::File.write(::File.join(dir, "config.json"), %({"model_type":"not_a_real_arch"}))
      ex = expect_raises(Exception) { SHAInet::HFLoader.load(dir) }
      ex.message.to_s.should contain("qwen3_5")
    ensure
      FileUtils.rm_rf(dir)
    end
  end
end
