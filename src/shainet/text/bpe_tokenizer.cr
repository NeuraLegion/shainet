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
    getter merges_rank : Hash(Tuple(String, String), Int32)

    def initialize
      @vocab = Hash(String, Int32).new
      @inv_vocab = [] of String
      @merges = [] of Tuple(String, String)
      @merges_map = Hash(Tuple(String, String), String).new
      @merges_rank = Hash(Tuple(String, String), Int32).new
    end

    # Train the tokenizer vocabulary from the given text using the
    # byte-pair encoding algorithm. `vocab_size` determines how many
    # unique tokens will be created at most.
    def train(text : String, vocab_size : Int32)
      word_freqs = Hash(String, Int32).new(0)
      text.split(/\s+/).each { |w| word_freqs[w] += 1 }

      corpus = Array(Array(String)).new(word_freqs.size)
      freqs = Array(Int32).new(word_freqs.size)
      word_freqs.each do |word, count|
        tokens = word.chars.map(&.to_s) + ["</w>"]
        corpus << tokens
        freqs << count
        tokens.each { |t| add_token(t) }
      end

      rounds = 0
      while @vocab.size < vocab_size
        rounds += 1
        if rounds % 100 == 0
          Log.debug { "Round #{rounds}: Merges done: #{@merges.size}, vocabulary size: #{@vocab.size}" }
          Log.debug { "Progress: #{(@vocab.size.to_f / vocab_size * 100.0).round(2)}%" }
        end
        pair_counts = Hash(Tuple(Int32, Int32), Int32).new(0)
        if CUDA.available? && CUDA.kernels_available?
          begin
            pairs_a = [] of Int32
            pairs_b = [] of Int32
            weights = [] of Int32
            corpus.each_with_index do |tokens, idx|
              freq = freqs[idx]
              (0...(tokens.size - 1)).each do |i|
                if id1 = @vocab[tokens[i]]?
                  if id2 = @vocab[tokens[i + 1]]?
                    pairs_a << id1
                    pairs_b << id2
                    weights << freq
                  end
                end
              end
            end
            if !pairs_a.empty?
              vocab_n = @vocab.size
              counts = Array(Int32).new(vocab_n*vocab_n, 0)
              counts_dev = Pointer(Int32).null
              a_dev = Pointer(Int32).null
              b_dev = Pointer(Int32).null
              f_dev = Pointer(Int32).null
              bytes_counts = (counts.size * 4).to_u64
              bytes_pairs = (pairs_a.size * 4).to_u64
              CUDA.malloc(pointerof(counts_dev).as(Pointer(Pointer(Void))), bytes_counts)
              CUDA.malloc(pointerof(a_dev).as(Pointer(Pointer(Void))), bytes_pairs)
              CUDA.malloc(pointerof(b_dev).as(Pointer(Pointer(Void))), bytes_pairs)
              CUDA.malloc(pointerof(f_dev).as(Pointer(Pointer(Void))), bytes_pairs)
              CUDA.memcpy(a_dev.as(Pointer(Void)), pairs_a.to_unsafe.as(Pointer(Void)), bytes_pairs, CUDA::MemcpyKind::HostToDevice)
              CUDA.memcpy(b_dev.as(Pointer(Void)), pairs_b.to_unsafe.as(Pointer(Void)), bytes_pairs, CUDA::MemcpyKind::HostToDevice)
              CUDA.memcpy(f_dev.as(Pointer(Void)), weights.to_unsafe.as(Pointer(Void)), bytes_pairs, CUDA::MemcpyKind::HostToDevice)
              CUDA.memcpy(counts_dev.as(Pointer(Void)), counts.to_unsafe.as(Pointer(Void)), bytes_counts, CUDA::MemcpyKind::HostToDevice)
              begin
                CUDA.count_token_pairs(counts_dev, a_dev, b_dev, f_dev, pairs_a.size, vocab_n)
                CUDA.memcpy(counts.to_unsafe.as(Pointer(Void)), counts_dev.as(Pointer(Void)), bytes_counts, CUDA::MemcpyKind::DeviceToHost)
                counts.each_with_index do |cnt, idx|
                  next if cnt == 0
                  id1 = idx // vocab_n
                  id2 = idx % vocab_n
                  pair_counts[{id1, id2}] = cnt
                end
              rescue
                pair_counts = cpu_pair_counts(corpus, freqs)
              end
              CUDA.free(a_dev.as(Pointer(Void)))
              CUDA.free(b_dev.as(Pointer(Void)))
              CUDA.free(f_dev.as(Pointer(Void)))
              CUDA.free(counts_dev.as(Pointer(Void)))
            end
          rescue
            pair_counts = cpu_pair_counts(corpus, freqs)
          end
        else
          pair_counts = cpu_pair_counts(corpus, freqs)
        end
        break if pair_counts.empty?
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
          size = tokens.size
          while i < size - 1
            if tokens[i] == token_a && tokens[i + 1] == token_b
              tokens[i] = new_token
              j = i + 1
              while j < size - 1
                tokens[j] = tokens[j + 1]
                j += 1
              end
              tokens.pop
              size -= 1
            else
              i += 1
            end
          end
        end
        pair = {token_a, token_b}
        @merges << pair
        @merges_map[pair] = new_token
        @merges_rank[pair] = @merges.size - 1
        add_token(new_token)
      end
    end

    # Encode a string into token IDs. Unknown tokens are added to the
    # vocabulary using a greedy BPE merging strategy.
    def encode(text : String) : Array(Int32)
      words = text.split(/\s+/)
      ids = Array(Int32).new(words.size)
      words.each do |word|
        tokens = encode_tokens(word)
        tokens.each { |t| ids << add_token(t) }
      end
      ids
    end

    private def encode_tokens(word : String) : Array(String)
      tokens = word.chars.map(&.to_s) + ["</w>"]
      return tokens if @merges_rank.empty?

      pairs = get_pairs(tokens)
      while !pairs.empty?
        best_pair = nil
        best_rank = Int32::MAX
        pairs.each do |p|
          if rank = @merges_rank[p]?
            if rank < best_rank
              best_rank = rank
              best_pair = p
            end
          end
        end
        break unless best_pair
        merge_tokens!(tokens, best_pair)
        break if tokens.size <= 1
        pairs = get_pairs(tokens)
      end

      tokens
    end

    private def get_pairs(tokens : Array(String)) : Array(Tuple(String, String))
      pairs = Array(Tuple(String, String)).new(tokens.size - 1)
      (0...tokens.size - 1).each do |i|
        pairs << {tokens[i], tokens[i + 1]}
      end
      pairs
    end

    # Decode an array of token IDs back into a string.
    def decode(ids : Array(Int32)) : String
      String.build do |io|
        current = String::Builder.new
        ids.each do |id|
          token = @inv_vocab[id]? || ""
          if token.ends_with?("</w>")
            current << token[0, token.size - 4]
            io << current.to_s
            io << ' '
            current = String::Builder.new
          else
            current << token
          end
        end
      end.rstrip
    end

    private def merge_tokens!(tokens : Array(String), pair : Tuple(String, String))
      new_token = @merges_map[pair]?
      new_token ||= String.build do |io|
        io << pair[0]
        io << pair[1]
      end
      i = 0
      size = tokens.size
      while i < size - 1
        if tokens[i] == pair[0] && tokens[i + 1] == pair[1]
          tokens[i] = new_token
          j = i + 1
          while j < size - 1
            tokens[j] = tokens[j + 1]
            j += 1
          end
          tokens.pop
          size -= 1
        else
          i += 1
        end
      end
    end

    private def cpu_pair_counts(corpus, freqs)
      pc = Hash(Tuple(Int32, Int32), Int32).new(0)
      corpus.each_with_index do |tokens, idx|
        freq = freqs[idx]
        (0...(tokens.size - 1)).each do |i|
          if id1 = @vocab[tokens[i]]?
            if id2 = @vocab[tokens[i + 1]]?
              pc[{id1, id2}] += freq
            end
          end
        end
      end
      pc
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
