require "./spec_helper"

# Which key head a value head reads is a property of the WEIGHT LAYOUT, not a free choice, so
# both conventions have to stay reachable and neither may silently become the other.
#
# grouped (k_head_tiled false) is the SafeTensors layout: HFLoader#split_head_interleaved
# rearranges the fused projection so the value heads sharing a key head land contiguously, which
# is what HF's repeat_interleave expects.
#
# tiled (k_head_tiled true) is the GGUF layout: llama.cpp widens q/k from num_k_heads to
# num_v_heads with ggml_repeat, and ggml_repeat TILES rather than interleaving. Feeding
# llama.cpp's own dumped q/k/v/beta into the delta rule reproduces its dumped attn_output at
# cosine 1.000000 and magnitude ratio 1.000000 for tiled, against 0.74 / 0.16 / 0.26 for grouped
# at layers 0, 10 and 21 of Qwen3.8-27B. Getting this wrong leaves the model fluent but wrong:
# it predicted "AB" where llama.cpp and Ollama predict "system".
describe SHAInet::GatedDeltaNetBlock do
  describe "#k_head_for" do
    it "defaults to grouped indexing, which the SafeTensors layout needs" do
      b = SHAInet::GatedDeltaNetBlock.new(16, 32, 6, 2, 4, 4, 4)
      b.k_head_tiled?.should be_false
      (0...6).map { |h| b.k_head_for(h, 3) }.should eq([0, 0, 0, 1, 1, 1])
    end

    it "uses tiled indexing when the weights come from GGUF" do
      b = SHAInet::GatedDeltaNetBlock.new(16, 32, 6, 2, 4, 4, 4)
      b.k_head_tiled = true
      (0...6).map { |h| b.k_head_for(h, 3) }.should eq([0, 1, 0, 1, 0, 1])
    end

    it "matches Qwen3.8-27B's 48 value heads over 16 key heads" do
      b = SHAInet::GatedDeltaNetBlock.new(64, 128, 48, 16, 8, 8, 4)
      b.k_head_tiled = true
      mapped = (0...48).map { |h| b.k_head_for(h, 3) }
      mapped[0, 16].should eq((0...16).to_a)
      mapped[16, 16].should eq((0...16).to_a)
      mapped[32, 16].should eq((0...16).to_a)
      # Every key head is read by exactly three value heads either way, so a count check alone
      # would NOT have caught this -- only the per-head identity does.
      (0...16).each { |kh| mapped.count(kh).should eq(3) }
    end

    it "never reads a key head that does not exist" do
      b = SHAInet::GatedDeltaNetBlock.new(64, 128, 48, 16, 8, 8, 4)
      [false, true].each do |tiled|
        b.k_head_tiled = tiled
        (0...48).each { |h| b.k_head_for(h, 3).should be < 16 }
      end
    end
  end
end
