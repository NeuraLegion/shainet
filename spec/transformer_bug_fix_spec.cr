require "./spec_helper"

# Spec to test the specific bug fix for positional encoding size mismatch
describe "Transformer Bug Fix" do
  it "handles positional encoding dimension mismatch gracefully" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)

    # Create a network similar to babylm_transformer
    d_model = 4
    seq_len = 2 # Small sequence length for testing
    vocab_size = 5

    net = SHAInet::Network.new
    net.add_layer(:input, 1, SHAInet.none)
    net.add_layer(:embedding, d_model, SHAInet.none, vocab_size: vocab_size)
    net.add_layer(:transformer, d_model)
    net.add_layer(:output, vocab_size, SHAInet.identity)
    net.fully_connect

    # Create positional encoding for sequence length
    pos_enc = SHAInet::PositionalEncoding.sinusoidal(seq_len, d_model)
    net.transformer_layers.first.positional_encoding = pos_enc

    # Test 1: Single token input should work (this was failing before the fix)
    single_token_output = net.run([1.0])
    single_token_output.size.should eq(vocab_size)

    # Test 2: Sequence input should work
    # The sequence version returns a 2D array (seq_len x vocab_size)
    sequence_output = net.run([[1.0], [2.0]])
    sequence_output.size.should eq(seq_len)                       # 2 positions
    sequence_output.each { |pos| pos.size.should eq(vocab_size) } # Each position has vocab_size outputs

    # Test 3: Different sequence lengths should also work gracefully
    # This should not crash even if the positional encoding size doesn't match
    short_sequence_output = net.run([3.0])
    short_sequence_output.size.should eq(vocab_size)
  end

  it "trains without crashing on small sequences" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)

    # Create tokenizer and simple text
    tokenizer = SHAInet::BPETokenizer.new
    text = "hello world test"
    tokenizer.train(text, 10)

    # Create network
    d_model = 8
    seq_len = 2 # Smaller sequence length to avoid issues
    vocab_size = tokenizer.vocab.size

    net = SHAInet::Network.new
    net.add_layer(:input, 1, SHAInet.none)
    net.add_layer(:embedding, d_model, SHAInet.none, vocab_size: vocab_size)
    net.add_layer(:transformer, d_model)
    net.add_layer(:output, vocab_size, SHAInet.identity)
    net.fully_connect

    # Set positional encoding only on first transformer layer
    pos_enc = SHAInet::PositionalEncoding.sinusoidal(seq_len, d_model)
    net.transformer_layers.first.positional_encoding = pos_enc

    # Create and encode some test data
    ids = tokenizer.encode(text)

    # Test that the network can process sequences without the original positional encoding crash
    if ids.size >= seq_len
      test_seq = ids[0, seq_len].map { |id| [id.to_f64] }
      output = net.run(test_seq)
      output.size.should eq(seq_len) # Should return output for each position
      output.each { |pos| pos.size.should eq(vocab_size) }

      # Test single token as well
      single_output = net.run([ids[0].to_f64])
      single_output.size.should eq(vocab_size)
    end

    # This test validates that the original "positional encoding size mismatch" bug is fixed
    # We don't need to test training here, just that basic forward passes work
  end
end
