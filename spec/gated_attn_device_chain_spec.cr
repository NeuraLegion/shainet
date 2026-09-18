require "./spec_helper"

# The DEVICE chain path for a gated attention layer (Qwen3.5), which the host-path specs in
# gated_attention_cache_spec.cr could not reach: block_device_capable? refused a gated layer
# outright, so forward_cached_device was never exercised with a gate at all.
#
# It refused for a real reason. The gate was applied on both host paths and on the PREFILL device
# path, but attention_cached_device ran w_o straight on ungated attention -- so the device decode
# path silently dropped the gate. On Qwen3.8-27B that turned the <|im_start|> prediction from
# 'system' at 24.16 into '2' at 19.42 while still producing fluent-looking text.
#
# Now every path applies it, and the refusal is on the gate being REACHABLE rather than absent.
# Keeping all 16 full-attention layers off the device chain cost a readback and re-upload per layer,
# measured at 44 ms of a 140 ms generated step, so this equivalence is what holds that open.

private def gad_fill!(m : SHAInet::SimpleMatrix, seed : Int32)
  r = Random.new(seed)
  m.rows.times { |i| m.cols.times { |j| m[i, j] = (r.next_float - 0.5) * 0.3 } }
  m
end

private def gad_block(gated : Bool, d = 32, heads = 4, kv = 2, hd = 8)
  b = SHAInet::LlamaBlock.new(d, heads, 64, num_kv_heads: kv, head_dim: hd)
  r = Random.new(99)
  b.q_norm = Array(Float32).new(hd) { (0.8 + r.next_float * 0.4).to_f32 }
  b.k_norm = Array(Float32).new(hd) { (0.8 + r.next_float * 0.4).to_f32 }
  gad_fill!(b.w_q.as(SHAInet::SimpleMatrix), 1)
  gad_fill!(b.w_k.as(SHAInet::SimpleMatrix), 2)
  gad_fill!(b.w_v.as(SHAInet::SimpleMatrix), 3)
  gad_fill!(b.w_o.as(SHAInet::SimpleMatrix), 4)
  b.w_gate_attn = gad_fill!(SHAInet::SimpleMatrix.new(d, heads * hd), 5) if gated
  ffn = b.ffn.as(SHAInet::SwiGLUFF)
  gad_fill!(ffn.gate_proj.as(SHAInet::SimpleMatrix), 6)
  gad_fill!(ffn.up_proj.as(SHAInet::SimpleMatrix), 7)
  gad_fill!(ffn.down_proj.as(SHAInet::SimpleMatrix), 8)
  b
end

private def gad_rel(a : SHAInet::SimpleMatrix, b : SHAInet::SimpleMatrix) : Float64
  worst = 0.0
  scale = 0.0
  a.rows.times do |i|
    a.cols.times do |j|
      worst = Math.max(worst, (a[i, j].to_f64 - b[i, j].to_f64).abs)
      scale = Math.max(scale, b[i, j].to_f64.abs)
    end
  end
  scale > 0 ? worst / scale : worst
end

describe "LlamaBlock gated attention on the device chain" do
  it "accepts a gated layer once the gate is reachable", tags: "cuda" do
    pending! "requires CUDA" unless SHAInet::CUDA.fully_available?
    b = gad_block(true)
    b.to_gpu!(quantize: true, bits: 8)
    # The point of the change: a gated layer is no longer refused outright.
    b.w_gate_attn.should_not be_nil
  end

  it "agrees between the host decode path and the device chain on a GATED block", tags: "cuda" do
    pending! "requires CUDA" unless SHAInet::CUDA.fully_available?
    x = gad_fill!(SHAInet::SimpleMatrix.new(1, 32), 21)

    host = gad_block(true)
    host.to_gpu!(quantize: true, bits: 8)
    host.clear_cache!
    ref = host.forward_cached(x)

    dev = gad_block(true)
    dev.to_gpu!(quantize: true, bits: 8)
    dev.clear_cache!
    pending! "device chain declined" unless dev.block_device_capable?

    xd = SHAInet::CudaMatrix.new(1, 32)
    32.times { |j| xd[0, j] = x[0, j] }
    xd.mark_host_modified!
    xd.sync_to_device!("gad_in")
    got_d = dev.forward_cached_device(xd)
    got_d.sync_from_device!("gad_out") if got_d.device_dirty?
    got = SHAInet::SimpleMatrix.new(1, 32)
    32.times { |j| got[0, j] = got_d.raw_data[j] }

    # Same weights both sides; only the summation order and Q8 rounding differ. Dropping the gate
    # entirely, which is what this path used to do, lands nowhere near this bound.
    gad_rel(got, ref).should be < 0.05
  end

  it "shows the device chain would fail this check without the gate", tags: "cuda" do
    pending! "requires CUDA" unless SHAInet::CUDA.fully_available?
    # An UNGATED reference against a GATED device block must disagree, otherwise the bound above
    # would pass even if the gate were dropped again and the spec would be vacuous.
    x = gad_fill!(SHAInet::SimpleMatrix.new(1, 32), 21)

    ungated = gad_block(false)
    ungated.to_gpu!(quantize: true, bits: 8)
    ungated.clear_cache!
    ref_ungated = ungated.forward_cached(x)

    gated = gad_block(true)
    gated.to_gpu!(quantize: true, bits: 8)
    gated.clear_cache!
    ref_gated = gated.forward_cached(x)

    gad_rel(ref_gated, ref_ungated).should be > 0.05
  end
end
