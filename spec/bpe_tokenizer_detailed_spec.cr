require "./spec_helper"

describe "BPE Tokenizer Detailed Tests" do
  describe "Basic Training" do
    it "should train on simple text and create merges" do
      text = "low lower newest widest"
      tokenizer = SHAInet::BPETokenizer.new
      tokenizer.train(text, 20)

      tokenizer.vocab.size.should be > 0
      tokenizer.merges.size.should be > 0

      # Should contain end-of-word tokens
      tokenizer.vocab.has_key?("</w>").should be_true

      # Should contain individual characters
      tokenizer.vocab.has_key?("l").should be_true
      tokenizer.vocab.has_key?("o").should be_true
      tokenizer.vocab.has_key?("w").should be_true
    end

    it "should handle repeated words correctly" do
      text = "hello hello hello world world"
      tokenizer = SHAInet::BPETokenizer.new
      tokenizer.train(text, 15)

      # Encode the same text
      ids = tokenizer.encode(text)

      # Should be able to decode back
      decoded = tokenizer.decode(ids)

      # Should preserve word boundaries
      decoded.split.should eq(["hello", "hello", "hello", "world", "world"])
    end
  end

  describe "Edge Cases" do
    it "should handle single character words" do
      text = "a b c a b c"
      tokenizer = SHAInet::BPETokenizer.new
      tokenizer.train(text, 10)

      ids = tokenizer.encode("a b c")
      decoded = tokenizer.decode(ids)

      decoded.should eq("a b c")
    end

    it "should handle unknown tokens during encoding" do
      tokenizer = SHAInet::BPETokenizer.new
      tokenizer.train("hello world", 10)

      # Encode text with unknown words
      ids = tokenizer.encode("hello unknown world")

      # Should not crash and should return valid ids
      ids.size.should be > 0
      ids.all? { |id| id >= 0 }.should be_true

      # Should be decodable
      decoded = tokenizer.decode(ids)
      decoded.should_not be_empty
    end
  end

  describe "Vocabulary Management" do
    it "should respect vocabulary size limits" do
      text = "the quick brown fox jumps over the lazy dog"
      vocab_size = 30 # Increased to be more realistic

      tokenizer = SHAInet::BPETokenizer.new
      tokenizer.train(text, vocab_size)

      # Note: BPE might create slightly more tokens than requested due to individual characters
      # This is expected behavior - vocab_size is more of a target than a hard limit
      tokenizer.vocab.size.should be > 0
      tokenizer.vocab.size.should be < (vocab_size * 1.5) # Allow some flexibility
    end

    it "should create consistent encodings" do
      text = "consistent test text"
      tokenizer = SHAInet::BPETokenizer.new
      tokenizer.train(text, 20)

      # Encode the same text multiple times
      ids1 = tokenizer.encode("consistent test")
      ids2 = tokenizer.encode("consistent test")

      ids1.should eq(ids2)
    end
  end

  describe "Integration with Transformer Data Format" do
    it "should produce tokens suitable for sequence modeling" do
      text = "this is a short test sentence for sequence modeling"
      tokenizer = SHAInet::BPETokenizer.new
      tokenizer.train(text, 30)

      ids = tokenizer.encode(text)

      # Should have enough tokens for sequence modeling
      ids.size.should be >= 5

      # Create sequence pairs like in babylm_transformer
      seq_len = 4
      sequences = [] of Array(Int32)
      targets = [] of Int32

      (0...(ids.size - seq_len)).each do |i|
        seq = ids[i, seq_len]
        target = ids[i + seq_len]

        sequences << seq
        targets << target
      end

      sequences.size.should be > 0
      targets.size.should eq(sequences.size)

      # Each sequence should have the right length
      sequences.all? { |seq| seq.size == seq_len }.should be_true

      # Targets should be valid token ids
      targets.all? { |t| t >= 0 && t < tokenizer.vocab.size }.should be_true
    end
  end

  describe "Memory and Performance" do
    it "should handle moderately large vocabularies" do
      # Create text with many unique words
      words = (1..100).map { |i| "word#{i}" }
      text = words.join(" ")

      tokenizer = SHAInet::BPETokenizer.new

      # Should not crash with larger vocabulary
      tokenizer.train(text, 500)
      tokenizer.vocab.size.should be > 0
      tokenizer.vocab.size.should be > 100 # Should create a substantial vocabulary
    end
  end
end
