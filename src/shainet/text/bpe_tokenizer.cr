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

      use_gpu = CUDA.available? && ENV["SHAINET_BPE_CUDA"]?

      while @vocab.size < vocab_size
        pair_list = [] of Tuple(Int32, Int32)
        corpus.each do |tokens|
          (0...(tokens.size - 1)).each do |i|
            id1 = @vocab[tokens[i]]
            id2 = @vocab[tokens[i + 1]]
            pair_list << {id1, id2}
          end
        end
        break if pair_list.empty?

        vsize = @vocab.size
        rows = pair_list.size
        product = rows.to_u64 * vsize.to_u64

        best_pair_ids = nil

        if use_gpu && product <= Int32::MAX.to_u64
          begin
            a = CudaMatrix.new(rows, vsize)
            b = CudaMatrix.new(rows, vsize)
            pair_list.each_with_index do |(id1, id2), idx|
              a[idx, id1] = 1.0
              b[idx, id2] = 1.0
            end
            a.sync_to_device!
            b.sync_to_device!
            counts = a.transpose * b
            counts.as?(CudaMatrix).try &.sync_from_device!
            best_pair_ids = pair_list.reduce(pair_list.first) do |best, pair|
              count = counts[pair[0], pair[1]].to_i
              best_count = counts[best[0], best[1]].to_i
              count > best_count ? pair : best
            end
          rescue e
            Log.warn { "Falling back to CPU: #{e.message}" }
          end
        end

        if best_pair_ids.nil?
          Log.warn { "Falling back to CPU: #{rows} × #{vsize} too large for GPU matrix." } if use_gpu && product > Int32::MAX.to_u64
          pair_counts = Hash(Tuple(String, String), Int32).new(0)
          corpus.each do |tokens|
            (0...(tokens.size - 1)).each do |i|
              pair = {tokens[i], tokens[i + 1]}
              pair_counts[pair] += 1
            end
          end
          break if pair_counts.empty?
          best_pair, _ = pair_counts.max_by { |_, count| count }
          best_pair_ids = {@vocab[best_pair[0]], @vocab[best_pair[1]]}
        end

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
