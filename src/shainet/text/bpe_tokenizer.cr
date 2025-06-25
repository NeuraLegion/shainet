require "../cuda"
require "../math/simple_matrix"
require "../math/cuda_matrix"

module SHAInet
  # Simple byte-pair encoding tokenizer. It can train a vocabulary
  # from text and encode/decode using the learned merges.
  class BPETokenizer
    Log = ::Log.for(self)
    getter vocab : Hash(String, Int32)
    getter inv_vocab : Array(String)
    getter merges : Array(Tuple(String, String))

    def initialize
      @vocab = Hash(String, Int32).new
      @inv_vocab = [] of String
      @merges = [] of Tuple(String, String)
    end

    # Train the tokenizer vocabulary from the given text using the
    # byte-pair encoding algorithm. `vocab_size` determines how many
    # unique tokens will be created at most.
    def train(text : String, vocab_size : Int32)
      corpus = text.split(/\s+/).map { |w| w.chars.map(&.to_s) + ["</w>"] }
      corpus.each do |tokens|
        tokens.each { |t| add_token(t) }
      end

      while @vocab.size < vocab_size
        Log.debug { "Merges done: #{@merges.size}, vocabulary size: #{@vocab.size}" }
        Log.debug { "Progress: #{(@vocab.size.to_f / vocab_size * 100.0).round(2)}%" }
        pair_counts = Hash(Tuple(Int32, Int32), Int32).new(0)
        corpus.each do |tokens|
          (0...(tokens.size - 1)).each do |i|
            id1 = @vocab[tokens[i]]
            id2 = @vocab[tokens[i + 1]]
            pair_counts[{id1, id2}] += 1
          end
        end
        break if pair_counts.empty?
        Log.debug { "Found #{pair_counts.size} unique pairs to merge." }
        best_pair_ids, _ = pair_counts.max_by { |_, count| count }

        token_a = @inv_vocab[best_pair_ids[0]]
        token_b = @inv_vocab[best_pair_ids[1]]
        new_token = token_a + token_b
        corpus.each do |tokens|
          i = 0
          while i < tokens.size - 1
            if tokens[i] == token_a && tokens[i + 1] == token_b
              tokens[i] = new_token
              tokens.delete_at(i + 1)
            else
              i += 1
            end
          end
        end
        @merges << {token_a, token_b}
        add_token(new_token)
      end
    end

    # Encode a string into token IDs. Unknown tokens are added to the
    # vocabulary.
    def encode(text : String) : Array(Int32)
      ids = [] of Int32
      text.split(/\s+/).each do |word|
        tokens = word.chars.map(&.to_s) + ["</w>"]
        @merges.each { |pair| merge_tokens!(tokens, pair) }
        tokens.each { |t| ids << add_token(t) }
      end
      ids
    end

    # Decode an array of token IDs back into a string.
    def decode(ids : Array(Int32)) : String
      words = [] of String
      current = ""
      ids.each do |id|
        token = @inv_vocab[id]? || ""
        if token.ends_with?("</w>")
          current += token.chomp("</w>")
          words << current
          current = ""
        else
          current += token
        end
      end
      words.join(" ")
    end

    private def merge_tokens!(tokens : Array(String), pair : Tuple(String, String))
      i = 0
      while i < tokens.size - 1
        if tokens[i] == pair[0] && tokens[i + 1] == pair[1]
          tokens[i] = pair[0] + pair[1]
          tokens.delete_at(i + 1)
        else
          i += 1
        end
      end
    end

    private def add_token(token : String) : Int32
      if id = @vocab[token]?
        id
      else
        new_id = @vocab.size
        @vocab[token] = new_id
        @inv_vocab << token
        new_id
      end
    end
  end
end
