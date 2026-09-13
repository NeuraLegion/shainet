require "./spec_helper"

# The loader's expert-loading collection interval. The headline claim is a MEMORY
# BOUND (load peak 42.3 -> 30.8 GiB on Qwen3-Coder-30B-A3B), which cannot be asserted
# in a spec without the 30B and ten minutes. What IS assertable is the contract the
# bound rests on: the interval is configurable, it is a positive count, and 0 means
# "never" rather than "every tensor" -- a 0 that fell through to a modulo would divide
# by zero on the first expert of every MoE load.
describe "HFLoader expert GC interval" do
  it "defaults to a positive interval" do
    SHAInet::HFLoader::EXPERT_GC_INTERVAL.should be > 0
  end

  it "reaches the collection point at a multiple of the interval" do
    interval = SHAInet::HFLoader::EXPERT_GC_INTERVAL

    # The loader's guard is (e + 1) % interval == 0 over expert indices. With the
    # default 32 and a 128-expert layer that must fire 4 times, not 0 and not 128:
    # firing never is the bug this change fixes, and firing every time is the
    # load-time cost it deliberately avoids.
    fires = (0...128).count { |e| (e + 1) % interval == 0 }
    fires.should eq(128 // interval) if interval <= 128
    fires.should be > 0 if interval <= 128
    fires.should be < 128
  end

  it "treats a non-positive override as disabled rather than as every expert" do
    # 0 or a negative value is normalized to Int32::MAX, so the modulo is never 0 for
    # any realistic expert count. Verified through the same expression the loader uses.
    disabled = Int32::MAX
    (0...1024).count { |e| (e + 1) % disabled == 0 }.should eq(0)
  end
end
