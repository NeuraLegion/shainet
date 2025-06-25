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
    getter merges_map : Hash(Tuple(String, String), String)

    def initialize
      @vocab = Hash(String, Int32).new
      @inv_vocab = [] of String
      @merges = [] of Tuple(String, String)
      @merges_map = Hash(Tuple(String, String), String).new
    end

    # Train the tokenizer vocabulary from the given text using the
    # byte-pair encoding algorithm. `vocab_size` determines how many
    # unique tokens will be created at most.
    def train(text : String, vocab_size : Int32)
      word_freqs = Hash(String, Int32).new(0)
      text.split(/\s+/).each { |w| word_freqs[w] += 1 }

      corpus = [] of Array(String)
      freqs = [] of Int32
      word_freqs.each do |word, count|
        tokens = word.chars.map(&.to_s) + ["</w>"]
        corpus << tokens
        freqs << count
        tokens.each { |t| add_token(t) }
      end

      while @vocab.size < vocab_size
        Log.debug { "Merges done: #{@merges.size}, vocabulary size: #{@vocab.size}" }
        Log.debug { "Progress: #{(@vocab.size.to_f / vocab_size * 100.0).round(2)}%" }
        pair_counts = Hash(Tuple(Int32, Int32), Int32).new(0)
        corpus.each_with_index do |tokens, idx|
          freq = freqs[idx]
          (0...(tokens.size - 1)).each do |i|
            t1 = tokens[i]
            t2 = tokens[i + 1]
            id1 = @vocab[t1]?
            id2 = @vocab[t2]?
            if id1 && id2
              pair_counts[{id1, id2}] += freq
            end
          end
        end
        break if pair_counts.empty?
        Log.debug { "Found #{pair_counts.size} unique pairs to merge." }
        heap = PairHeap.new
        pair_counts.each do |pair, count|
          heap.push PairHeap::Node.new(pair, count)
        end
        best_node = heap.pop
        break unless best_node
        best_pair_ids = best_node.pair

        token_a = @inv_vocab[best_pair_ids[0]]
        token_b = @inv_vocab[best_pair_ids[1]]
        new_token = String.build do |io|
          io << token_a
          io << token_b
        end
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
        pair = {token_a, token_b}
        @merges << pair
        @merges_map[pair] = new_token
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
      new_token = @merges_map[pair]?
      new_token ||= String.build do |io|
        io << pair[0]
        io << pair[1]
      end
      i = 0
      while i < tokens.size - 1
        if tokens[i] == pair[0] && tokens[i + 1] == pair[1]
          tokens[i] = new_token
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

  # Simple max-heap used to select the best pair during training
  class PairHeap
    struct Node
      getter pair : Tuple(Int32, Int32)
      getter count : Int32

      def initialize(@pair : Tuple(Int32, Int32), @count : Int32)
      end
    end

    def initialize
      @data = [] of Node
    end

    def push(node : Node)
      @data << node
      sift_up(@data.size - 1)
    end

    def pop : Node?
      return nil if @data.empty?
      max = @data[0]
      last = @data.pop
      if !@data.empty?
        @data[0] = last
        sift_down(0)
      end
      max
    end

    private def sift_up(idx : Int32)
      while idx > 0
        parent = (idx - 1) // 2
        if @data[parent].count < @data[idx].count
          @data[parent], @data[idx] = @data[idx], @data[parent]
          idx = parent
        else
          break
        end
      end
    end

    private def sift_down(idx : Int32)
      size = @data.size
      while true
        left = idx * 2 + 1
        right = left + 1
        largest = idx
        largest = left if left < size && @data[left].count > @data[largest].count
        largest = right if right < size && @data[right].count > @data[largest].count
        break if largest == idx
        @data[idx], @data[largest] = @data[largest], @data[idx]
        idx = largest
      end
    end
  end
end
