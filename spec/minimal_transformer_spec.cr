require "./spec_helper"

describe "Minimal Transformer Pipeline" do
  it "should train a minimal transformer without crashing" do
    # Very simple test data
    text = "cat dog cat dog"
    vocab_size = 10
    seq_len = 2
    d_model = 4

    # Train tokenizer
    tokenizer = SHAInet::BPETokenizer.new
    tokenizer.train(text, vocab_size)
    ids = tokenizer.encode(text)

    # Should have at least seq_len + 1 tokens for training
    if ids.size <= seq_len
      pending "Not enough tokens for sequence training"
    end

    # Build minimal network
    token_count = tokenizer.vocab.size

    net = SHAInet::Network.new
    net.add_layer(:input, 1, :memory, SHAInet.none)
    net.add_layer(:embedding, d_model, :memory, SHAInet.none, vocab_size: token_count)
    net.add_layer(:transformer, d_model, num_heads: 1, ff_hidden: 8)
    net.add_layer(:output, token_count, :memory, SHAInet.identity)
    net.fully_connect

    # Apply positional encoding to first transformer layer only
    pos_enc = SHAInet::PositionalEncoding.sinusoidal(seq_len, d_model)
    net.transformer_layers.first.positional_encoding = pos_enc

    # Test forward pass with single token first (simpler case)
    net.learning_rate = 0.01

    begin
      # Test with single token input (should work without positional encoding issues)
      single_output = net.run([ids.first.to_f64])
      single_output.size.should eq(token_count)

      # Now test with sequence input
      seq_input = ids[0, seq_len].map { |id| [id.to_f64] }

      # For language modeling, we typically want just the last position output
      sequence_outputs = net.run(seq_input)

      # The sequence output should be a 2D array: [seq_len, vocab_size]
      sequence_outputs.should be_a(Array(Array(Float64)))
      sequence_outputs.as(Array(Array(Float64))).size.should eq(seq_len)
      sequence_outputs.as(Array(Array(Float64))).each do |output|
        output.size.should eq(token_count)
      end

      # For next-token prediction, we typically use the last position
      last_output = sequence_outputs.as(Array(Array(Float64))).last
      last_output.size.should eq(token_count)

      # If we get here, the basic pipeline works
      true.should be_true
    rescue e : Exception
      # If it crashes, we want to see the error
      fail "Pipeline crashed with: #{e.message}"
    end
  end

  it "should handle different sequence lengths correctly" do
    # Test with different sequence lengths to ensure positional encoding works
    text = "one two three four five six"
    tokenizer = SHAInet::BPETokenizer.new
    tokenizer.train(text, 15)
    ids = tokenizer.encode(text)

    if ids.size < 4
      pending "Not enough tokens for test"
    end

    d_model = 4

    [2, 3].each do |seq_len|
      next if ids.size <= seq_len

      token_count = tokenizer.vocab.size
      net = SHAInet::Network.new
      net.add_layer(:input, 1, :memory, SHAInet.none)
      net.add_layer(:embedding, d_model, :memory, SHAInet.none, vocab_size: token_count)
      net.add_layer(:transformer, d_model, num_heads: 1, ff_hidden: 8)
      net.add_layer(:output, token_count, :memory, SHAInet.identity)
      net.fully_connect

      # Create positional encoding for this sequence length
      pos_enc = SHAInet::PositionalEncoding.sinusoidal(seq_len, d_model)
      net.transformer_layers.first.positional_encoding = pos_enc

      # Test sequence of this length
      seq = ids[0, seq_len].map { |id| [id.to_f64] }

      begin
        sequence_outputs = net.run(seq)
        sequence_outputs.should be_a(Array(Array(Float64)))
        sequence_outputs.as(Array(Array(Float64))).size.should eq(seq_len)
        sequence_outputs.as(Array(Array(Float64))).each do |output|
          output.size.should eq(token_count)
        end
      rescue e : Exception
        fail "Failed with sequence length #{seq_len}: #{e.message}"
      end
    end
  end

  it "should handle dimension mismatches gracefully" do
    # Test that our fix handles dimension mismatches gracefully
    d_model = 4

    net = SHAInet::Network.new
    net.add_layer(:input, 1, :memory, SHAInet.none)
    net.add_layer(:embedding, d_model, :memory, SHAInet.none, vocab_size: 10)
    net.add_layer(:transformer, d_model, num_heads: 1, ff_hidden: 8)
    net.add_layer(:output, 10, :memory, SHAInet.identity)
    net.fully_connect

    # Create positional encoding with sequence length 5
    long_seq_len = 5
    pos_enc = SHAInet::PositionalEncoding.sinusoidal(long_seq_len, d_model)
    net.transformer_layers.first.positional_encoding = pos_enc

    # Try to run with different sequence lengths - should work now
    [1, 2, 3, 4, 5, 6].each do |actual_seq_len|
      seq = (0...actual_seq_len).map { |i| [i.to_f64] }

      begin
        result = net.run(seq)
        result.should be_a(Array(Array(Float64)))
        result.as(Array(Array(Float64))).size.should eq(actual_seq_len)
      rescue e : Exception
        fail "Failed with sequence length #{actual_seq_len}: #{e.message}"
      end
    end
  end
end
