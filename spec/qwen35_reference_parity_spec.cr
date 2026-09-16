require "./spec_helper"

# Parity against the real `transformers` implementation, captured by running
# Qwen3_5ForConditionalGeneration on CPU with output_hidden_states and forward hooks on layer 0.
#
# These numbers are an EXTERNAL oracle, not this codebase's own output, which is what makes them
# worth pinning. Two bugs were found by diffing against them and each was invisible to every other
# check available:
#
#   1. Qwen3_5RMSNorm stores gamma as an OFFSET FROM ONE (weight initialized to zeros, applied as
#      `x * (1.0 + weight)`). Read as a plain multiplier, layer 0's input_layernorm produced rms
#      0.116 instead of 1.090 -- a 9.4x error at the first norm of the first layer, compounding
#      through 32 layers with 4-5 norms each. Qwen3_5RMSNormGated is the exception (initialized to
#      ones, applied directly), so linear_attn.norm must NOT be offset.
#
#   2. The delta rule scales q by 1/sqrt(head_k). I had reasoned this was removed by the per-head
#      output norm; it is not, because RMSNorm is scale-invariant only while variance dominates its
#      epsilon, and here the core output's variance (~1.9e-6) is only about twice eps (1e-6).
#
# A shape check cannot catch either, and neither can an internally-consistent reimplementation:
# a numpy transcription written from the same misreading agreed with SHAInet exactly, at every
# stage, while both were wrong. Only the external oracle separated them.
#
# Skipped unless the checkpoint is present, since it is 19 GB and not in CI.
#
# TAGGED `perf`, so `crystal spec --tag '~perf'` excludes it. Not a nicety: the first example loads
# a layer in fp32 and the second loads all 32 layers, which took this host to 37.8 GB resident
# inside an ordinary suite run, on a 62 GB machine with no swap. Run it deliberately with
# `crystal spec --tag perf`.

MODEL_DIR       = "/home/unshadow/models/Qwen3.5-9B"
ORACLE_TOKENS   = [760, 6511, 314, 9338, 369] # "The capital of France is"
ORACLE_NORM1    = 1.090495
ORACLE_LAYER0   = 0.112054
ORACLE_L0_FIRST = [-0.05391, -0.03613, 0.00042, -0.02329, -0.10064, -0.01207]

def oracle_rms(m : SHAInet::SimpleMatrix) : Float64
  sum = 0.0
  m.rows.times { |i| m.cols.times { |j| v = m[i, j].to_f64; sum += v * v } }
  Math.sqrt(sum / (m.rows * m.cols))
end

describe "Qwen3.5 parity with the transformers reference", tags: "perf" do
  it "reproduces layer 0's input_layernorm and output" do
    pending! "checkpoint not present" unless ::File.exists?(::File.join(MODEL_DIR, "config.json"))

    net = SHAInet::HFLoader.load_qwen35(MODEL_DIR, 1, quantize: false)
    emb = net.hidden_layers.find(&.is_a?(SHAInet::EmbeddingLayer)).as(SHAInet::EmbeddingLayer)
    d = emb.embeddings.cols
    x = SHAInet::SimpleMatrix.new(ORACLE_TOKENS.size, d)
    ORACLE_TOKENS.each_with_index { |t, i| d.times { |j| x[i, j] = emb.embeddings[t, j] } }

    block = net.hidden_layers.reject(SHAInet::EmbeddingLayer).first.as(SHAInet::GatedDeltaNetBlock)

    # The norm alone, because reading gamma as a plain multiplier fails HERE, before any of the
    # mixer runs, and localizing it matters more than the layer output agreeing.
    oracle_rms(block.norm1.forward(x)).should be_close(ORACLE_NORM1, 1e-4)

    block.clear_cache!
    got = block.forward(x)
    oracle_rms(got).should be_close(ORACLE_LAYER0, 1e-4)
    ORACLE_L0_FIRST.each_with_index do |want, j|
      got[0, j].to_f64.should be_close(want, 5e-5)
    end
  end

  it "predicts Paris for the capital of France" do
    pending! "checkpoint not present" unless ::File.exists?(::File.join(MODEL_DIR, "config.json"))
    pending! "requires CUDA for a Q4 load" unless SHAInet::CUDA.fully_available?

    # The end-to-end claim, asserted as the top token rather than as a loss, so it cannot pass on a
    # model that merely learned token frequencies. The reference scores " Paris" at 17.117 in fp32;
    # Q4 here gives 16.98.
    net = SHAInet::HFLoader.load_qwen35(MODEL_DIR, nil, quantize: true, bits: 4)
    tok = SHAInet::BPETokenizer.from_hf(::File.join(MODEL_DIR, "tokenizer.json"))
    net.use_kv_cache = false
    m = net.run(tok.encode("The capital of France is"), stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
    r = m.rows - 1
    best = (0...m.cols).max_by { |j| m[r, j].to_f64 }
    tok.decode([best]).should eq(" Paris")
    m[r, best].to_f64.should be_close(17.0, 1.5)
  end
end
