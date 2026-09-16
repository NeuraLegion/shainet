require "./spec_helper"

# A kv-cached prefill must produce the same activations as an uncached forward on the same
# tokens. That equivalence is what lets generation reuse the cache at all, and nothing else
# checks it for a GATED attention layer (Qwen3.5), where q_proj carries a fused output gate.
#
# Measured motivation: on Qwen3.5-9B the cached path collapsed to the four most frequent tokens
# (" ", ",", ".", "\n") with logits near 14.4 while the uncached path gave rms 1.05 on the same
# prompt. Each looked plausible in isolation; only comparing them showed one was wrong.

def cga_fill!(m : SHAInet::SimpleMatrix, seed : Int32)
  r = Random.new(seed)
  m.rows.times { |i| m.cols.times { |j| m[i, j] = (r.next_float - 0.5) * 0.3 } }
  m
end

def cga_block(gated : Bool, d = 32, heads = 4, kv = 2, hd = 8, qk_norm : Bool = true)
  b = SHAInet::LlamaBlock.new(d, heads, 64, num_kv_heads: kv, head_dim: hd)
  if qk_norm
    # Qwen3 and Qwen3.5 both carry per-head q_norm / k_norm. Leaving them nil made the first
    # version of this file pass while the real model's cached path diverged.
    r = Random.new(99)
    b.q_norm = Array(Float32).new(hd) { (0.8 + r.next_float * 0.4).to_f32 }
    b.k_norm = Array(Float32).new(hd) { (0.8 + r.next_float * 0.4).to_f32 }
  end
  cga_fill!(b.w_q.as(SHAInet::SimpleMatrix), 1)
  cga_fill!(b.w_k.as(SHAInet::SimpleMatrix), 2)
  cga_fill!(b.w_v.as(SHAInet::SimpleMatrix), 3)
  cga_fill!(b.w_o.as(SHAInet::SimpleMatrix), 4)
  if gated
    g = SHAInet::SimpleMatrix.new(d, heads * hd)
    b.w_gate_attn = cga_fill!(g, 5)
  end
  ffn = b.ffn.as(SHAInet::SwiGLUFF)
  cga_fill!(ffn.gate_proj.as(SHAInet::SimpleMatrix), 6)
  cga_fill!(ffn.up_proj.as(SHAInet::SimpleMatrix), 7)
  cga_fill!(ffn.down_proj.as(SHAInet::SimpleMatrix), 8)
  b
end

def cga_worst(a : SHAInet::SimpleMatrix, b : SHAInet::SimpleMatrix) : Float64
  worst = 0.0
  a.rows.times { |i| a.cols.times { |j| worst = Math.max(worst, (a[i, j].to_f64 - b[i, j].to_f64).abs) } }
  worst
end

# Peak magnitude, so the bounds below are RELATIVE.
#
# SimpleMatrix is Float32 and the two paths sum in different orders, so an absolute bound is
# arbitrary: at d=32 with QK-norm the honest disagreement measured 1.2e-4, and tightening the
# bound to exclude it would only be hiding rounding. Scaling by the activation's own magnitude
# keeps the check meaningful without inviting a nudge whenever a spec fails.
def cga_scale(m : SHAInet::SimpleMatrix) : Float64
  s = 0.0
  m.rows.times { |i| m.cols.times { |j| s = Math.max(s, m[i, j].to_f64.abs) } }
  s
end

CGA_REL = 1e-3

describe "LlamaBlock prefill / decode agreement" do
  it "agrees between forward and a cached prefill on an ungated block" do
    # The control: if this fails the harness is wrong, not the gate.
    x = cga_fill!(SHAInet::SimpleMatrix.new(5, 32), 20)
    ref = cga_block(false).forward(x)
    b = cga_block(false)
    b.clear_cache!
    got = b.forward_cached(x)
    (cga_worst(got, ref) / cga_scale(ref)).should be < CGA_REL
  end

  it "agrees between forward and a cached prefill on a GATED block" do
    x = cga_fill!(SHAInet::SimpleMatrix.new(5, 32), 20)
    ref = cga_block(true).forward(x)
    b = cga_block(true)
    b.clear_cache!
    got = b.forward_cached(x)
    (cga_worst(got, ref) / cga_scale(ref)).should be < CGA_REL
  end

  it "agrees between a whole-sequence forward and token-at-a-time decode on a GATED block" do
    # The path generation actually uses: prefill n-1 tokens, then step the last one, and the
    # final row must match the whole-sequence result. This is what catches a gate applied at the
    # wrong point in the cached step, or a position index off by one.
    x = cga_fill!(SHAInet::SimpleMatrix.new(4, 32), 21)
    ref = cga_block(true).forward(x)

    b = cga_block(true)
    b.clear_cache!
    head = SHAInet::SimpleMatrix.new(3, 32)
    3.times { |i| 32.times { |j| head[i, j] = x[i, j] } }
    b.forward_cached(head)
    last = SHAInet::SimpleMatrix.new(1, 32)
    32.times { |j| last[0, j] = x[3, j] }
    got = b.forward_cached(last)

    got.rows.should eq(1)
    worst = 0.0
    32.times { |j| worst = Math.max(worst, (got[0, j].to_f64 - ref[3, j].to_f64).abs) }
    (worst / cga_scale(ref)).should be < CGA_REL
  end

  it "agrees between forward and a cached prefill on a QUANTIZED gated block" do
    pending! "requires CUDA" unless SHAInet::CUDA.fully_available?
    # The real model is Q4, and the fp32 examples above passed while the 9B's first attention
    # layer diverged 71% between the two paths. Quantization is the variable they did not cover.
    x = cga_fill!(SHAInet::SimpleMatrix.new(5, 32), 20)
    ref = cga_block(true).forward(x)

    b = cga_block(true)
    b.to_gpu!(quantize: true, bits: 4)
    b.clear_cache!
    got = b.forward_cached(x)

    # Looser than CGA_REL because Q4 is lossy, but far tighter than the 0.71 the real model showed.
    (cga_worst(got, ref) / cga_scale(ref)).should be < 0.15
  end

  it "agrees between a quantized forward and a quantized cached prefill" do
    pending! "requires CUDA" unless SHAInet::CUDA.fully_available?
    # Both sides quantized, so quantization error cancels and only a PATH difference remains.
    # This is the one that isolates the bug from the lossiness.
    x = cga_fill!(SHAInet::SimpleMatrix.new(5, 32), 20)

    a = cga_block(true)
    a.to_gpu!(quantize: true, bits: 4)
    a.clear_cache!
    ref = a.forward(x)

    b = cga_block(true)
    b.to_gpu!(quantize: true, bits: 4)
    b.clear_cache!
    got = b.forward_cached(x)

    (cga_worst(got, ref) / cga_scale(ref)).should be < CGA_REL
  end

  it "shows the gate actually changes the output" do
    # The other direction: without this, a w_gate_attn that was silently ignored on BOTH paths
    # would make every example above pass.
    x = cga_fill!(SHAInet::SimpleMatrix.new(5, 32), 20)
    plain = cga_block(false).forward(x)
    gated = cga_block(true).forward(x)
    (cga_worst(plain, gated) / cga_scale(plain)).should be > CGA_REL
  end
end
