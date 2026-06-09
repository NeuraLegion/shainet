require "./spec_helper"
require "../src/shainet"

describe "HFLoader" do
  it "loads tiny-gpt2 from safetensors" do
    model_dir = File.join(__DIR__, "test_data/tiny-gpt2")
    pending!("tiny-gpt2 fixture missing") unless File.exists?(File.join(model_dir, "model.safetensors"))

    net = SHAInet::HFLoader.load_gpt2(model_dir)

    # Verify structure
    net.transformer_layers.size.should eq(5)

    # Check first transformer block
    t0 = net.transformer_layers[0]
    t0.mha.num_heads.should eq(4)
    t0.mha.d_model.should eq(32)

    # Verify weights were loaded (not all zeros)
    w_q = t0.mha.w_q
    case w_q
    when SHAInet::SimpleMatrix
      nonzero = (0...w_q.rows).any? { |r| (0...w_q.cols).any? { |c| w_q[r, c] != 0.0 } }
    when SHAInet::CudaMatrix
      w_q.sync_from_device!("test") if w_q.device_dirty?
      nonzero = (0...w_q.rows).any? { |r| (0...w_q.cols).any? { |c| w_q[r, c] != 0.0 } }
    else
      nonzero = false
    end
    nonzero.should be_true
  end

  it "produces output logits from forward pass" do
    model_dir = File.join(__DIR__, "test_data/tiny-gpt2")
    pending!("tiny-gpt2 fixture missing") unless File.exists?(File.join(model_dir, "model.safetensors"))

    net = SHAInet::HFLoader.load_gpt2(model_dir)

    # Input: single token [42] as column vector
    input = SHAInet::SimpleMatrix.new(1, 1)
    input[0, 0] = 42.0

    output = net.run(input)

    # Output should be [1, 1000] (vocab_size logits)
    output.rows.should eq(1)
    output.cols.should eq(1000)

    # Verify output is not all zeros (model actually computed something)
    has_nonzero = (0...output.cols).any? { |c| output[0, c] != 0.0 }
    has_nonzero.should be_true

    # Verify output has reasonable values (not NaN/Inf)
    all_finite = (0...output.cols).all? { |c| output[0, c].finite? }
    all_finite.should be_true
  end
end
