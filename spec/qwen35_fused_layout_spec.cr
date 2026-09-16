require "./spec_helper"

# The fused projections in Qwen3.5 are interleaved PER HEAD, and a contiguous split produces
# tensors of exactly the right shape holding scrambled contents. No shape assertion can catch
# that, and the only symptom is that the whole model sits at chance -- measured 12.19 nats
# against ln(248320) = 12.42. So the layout itself needs pinning, with values chosen so a
# contiguous split would demonstrably fail.

describe "HFLoader fused projection layouts" do
  it "deinterleaves a per-head [q | gate] projection" do
    # 2 heads, head_dim 3, so the 12 columns run:
    #   h0_q(0,1,2) h0_gate(3,4,5) h1_q(6,7,8) h1_gate(9,10,11)
    # A contiguous halving would put columns 0..5 in q, which mixes head 0's gate into it.
    src = SHAInet::SimpleMatrix.new(1, 12)
    12.times { |j| src[0, j] = j.to_f32 }

    q, g = SHAInet::HFLoader.split_head_interleaved(src, 2, 3)
    q.cols.should eq(6)
    g.cols.should eq(6)
    (0...6).map { |j| q[0, j].to_i }.should eq([0, 1, 2, 6, 7, 8])
    (0...6).map { |j| g[0, j].to_i }.should eq([3, 4, 5, 9, 10, 11])
  end

  it "refuses a projection whose width is not heads * 2 * head_dim" do
    src = SHAInet::SimpleMatrix.new(1, 11)
    expect_raises(Exception, /is not 2 \* 2 \* 3/) do
      SHAInet::HFLoader.split_head_interleaved(src, 2, 3)
    end
  end

  it "deinterleaves a fused qkv projection grouped per key head" do
    # 2 key heads, head_k 2, head_v 2, 2 value heads per key head, so the stride is
    # 2 + 2 + 4 = 8 and the 16 columns run:
    #   kh0: q(0,1) k(2,3) v(4,5,6,7)   kh1: q(8,9) k(10,11) v(12,13,14,15)
    src = SHAInet::SimpleMatrix.new(1, 16)
    16.times { |j| src[0, j] = j.to_f32 }

    q, k, v = SHAInet::HFLoader.split_qkv_interleaved(src, 2, 2, 2, 2)
    q.cols.should eq(4)
    k.cols.should eq(4)
    v.cols.should eq(8)
    (0...4).map { |j| q[0, j].to_i }.should eq([0, 1, 8, 9])
    (0...4).map { |j| k[0, j].to_i }.should eq([2, 3, 10, 11])
    (0...8).map { |j| v[0, j].to_i }.should eq([4, 5, 6, 7, 12, 13, 14, 15])
  end

  it "keeps value heads grouped under the key head the mixer will pair them with" do
    # The mixer indexes key head as h // heads_per_k, so value heads 0 and 1 must come from key
    # head 0's slice. Pinned separately because getting the columns right while grouping them
    # wrongly would still satisfy the element check above on a symmetric example.
    src = SHAInet::SimpleMatrix.new(1, 16)
    16.times { |j| src[0, j] = j.to_f32 }
    _, _, v = SHAInet::HFLoader.split_qkv_interleaved(src, 2, 2, 2, 2)
    # value head 1 (columns 2..3 of v) belongs to key head 0, so it holds 6,7 not 12,13.
    v[0, 2].to_i.should eq(6)
    v[0, 3].to_i.should eq(7)
  end

  it "deinterleaves the conv1d channels the same way, over rows" do
    src = SHAInet::SimpleMatrix.new(16, 2)
    16.times { |i| 2.times { |j| src[i, j] = (i * 10 + j).to_f32 } }
    q, k, v = SHAInet::HFLoader.split_qkv_interleaved_rows(src, 2, 2, 2, 2)
    q.rows.should eq(4)
    k.rows.should eq(4)
    v.rows.should eq(8)
    (0...4).map { |i| (q[i, 0].to_i / 10) }.should eq([0, 1, 8, 9])
    (0...4).map { |i| (k[i, 0].to_i / 10) }.should eq([2, 3, 10, 11])
    (0...8).map { |i| (v[i, 0].to_i / 10) }.should eq([4, 5, 6, 7, 12, 13, 14, 15])
  end

  it "reverses conv taps, translating PyTorch's order into ShortConv's" do
    # PyTorch's w[kernel-1] is the current position; ShortConv's w[0] is. Copying straight across
    # runs the convolution time-reversed, which is invisible to every shape check.
    src = SHAInet::SimpleMatrix.new(2, 4)
    2.times { |i| 4.times { |j| src[i, j] = (i * 4 + j).to_f32 } }
    dst = SHAInet::SimpleMatrix.new(2, 4)
    SHAInet::HFLoader.reverse_taps!(src, dst)
    (0...4).map { |j| dst[0, j].to_i }.should eq([3, 2, 1, 0])
    (0...4).map { |j| dst[1, j].to_i }.should eq([7, 6, 5, 4])
  end

  it "matches the real Qwen3.5-9B arithmetic, which is what identifies the layout" do
    # 16 key heads * (128 + 128 + 2 * 128) = 8192, exactly in_proj_qkv's output width. Three
    # contiguous blocks give the same total, so this equality is the evidence for the layout
    # rather than a coincidence to rely on.
    k_heads, head_k, head_v, heads_per_k = 16, 128, 128, 2
    (k_heads * (2 * head_k + heads_per_k * head_v)).should eq(8192)
    # and the gated attention head: 16 heads * 2 * 256 = 8192 as well.
    (16 * 2 * 256).should eq(8192)
  end
end
