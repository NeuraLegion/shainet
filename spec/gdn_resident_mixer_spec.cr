require "./spec_helper"

# The device-resident mixer replaces the host path's per-projection sync and its element-by-element
# conv / SiLU / gate / per-head-norm stages with device kernels. It must agree with the host path,
# which stays the reference implementation -- a silent divergence here is exactly the failure mode
# that leaves generation fluent but wrong, and it already happened once in this work: the alpha and
# beta projections were issued with the wrong transpose, which mixed the head and model axes and
# moved the <|im_start|> logit by 4% while still producing plausible text.
#
# SHAINET_GDN_RESIDENT=0 selects the host path, so both run in one process on identical weights.
describe "Gated DeltaNet resident mixer" do
  it "agrees with the host mixer", tags: "cuda" do
    pending! "no CUDA" unless SHAInet::CUDA.fully_available?
    pending! "no GDN kernels" unless SHAInet::CUDA.gated_delta_rule_available?
    pending! "no mixer kernels" unless SHAInet::CUDA.gdn_mixer_kernels_available?

    d = 64
    v_heads = 6
    k_heads = 2
    hk = 8
    hv = 8
    blk = SHAInet::GatedDeltaNetBlock.new(d, 2 * d, v_heads, k_heads, hk, hv, 4)

    fill = ->(m : SHAInet::SimpleMatrix, scale : Float64, phase : Float64) do
      m.rows.times { |i| m.cols.times { |j| m[i, j] = (scale * Math.sin(phase + (i * 7 + j * 13) * 0.37)).to_f32 } }
    end
    fill.call(blk.w_q.as(SHAInet::SimpleMatrix), 0.09, 0.1)
    fill.call(blk.w_k.as(SHAInet::SimpleMatrix), 0.08, 0.2)
    fill.call(blk.w_v.as(SHAInet::SimpleMatrix), 0.07, 0.3)
    fill.call(blk.w_o.as(SHAInet::SimpleMatrix), 0.06, 0.4)
    fill.call(blk.w_gate.as(SHAInet::SimpleMatrix), 0.05, 0.5)
    fill.call(blk.w_alpha.as(SHAInet::SimpleMatrix), 0.04, 0.6)
    fill.call(blk.w_beta.as(SHAInet::SimpleMatrix), 0.03, 0.7)
    fill.call(blk.conv_q.weight, 0.3, 0.8)
    fill.call(blk.conv_k.weight, 0.3, 0.9)
    fill.call(blk.conv_v.weight, 0.3, 1.0)
    v_heads.times { |h| blk.a_log[h] = -2.0 - 0.1 * h; blk.dt_bias[h] = 0.05 * h }
    fill.call(blk.out_norm.gamma.as(SHAInet::SimpleMatrix), 1.0, 0.2)

    seq = 5
    x = SHAInet::SimpleMatrix.new(seq, d)
    fill.call(x, 0.5, 1.5)

    # Quantize FIRST, then run both paths on the SAME weights, so the only difference under test is
    # the computation. Comparing fp32-host against quantized-device instead conflates the two and
    # measured 0.982 from quantization alone.
    blk.to_gpu!(quantize: true, bits: 8)

    ENV["SHAINET_GDN_RESIDENT"] = "0"
    blk.clear_cache!
    host = blk.mix(x.clone)

    ENV["SHAINET_GDN_RESIDENT"] = "1"
    blk.clear_cache!
    dev = blk.mix(x.clone)
    ENV.delete("SHAINET_GDN_RESIDENT")

    # Same weights both sides, so only summation order differs: a wrong axis, tap order or head map
    # destroys the cosine, fp32 reassociation does not.
    dot = 0.0
    nh = 0.0
    nd = 0.0
    seq.times do |t|
      d.times do |j|
        a = host[t, j].to_f64
        b = dev[t, j].to_f64
        dot += a * b
        nh += a * a
        nd += b * b
      end
    end
    cos = dot / (Math.sqrt(nh) * Math.sqrt(nd) + 1e-30)
    cos.should be > 0.999
  end
end
