# No CUDA imports needed - CPU-only implementation
require "json"

module SHAInet
  # Simple byte-pair encoding tokenizer. It can train a vocabulary
  # from text and encode/decode using the learned merges.
  class BPETokenizer
    Log = ::Log.for(self)
    property vocab : Hash(String, Int32)
    property inv_vocab : Array(String)
    property merges : Array(Tuple(String, String))
    property merges_map : Hash(Tuple(String, String), String)
    property merges_rank : Hash(Tuple(String, String), Int32)
    property hf_mode : Bool = false

    # GPT-2 byte-to-unicode mapping (lazily built). Maps a printable Unicode
    # codepoint back to the original raw byte it represents in byte-level BPE.
    @@unicode_to_byte : Hash(Char, UInt8)?
    @@byte_to_unicode : Hash(UInt8, Char)?

    def self.unicode_to_byte(ch : Char) : UInt8
      map = (@@unicode_to_byte ||= build_byte_map)
      map[ch]? || 0_u8
    end

    def self.byte_to_unicode(b : UInt8) : Char
      map = (@@byte_to_unicode ||= build_byte_to_unicode_map)
      map[b]
    end

    private def self.build_byte_to_unicode_map : Hash(UInt8, Char)
      # Inverse of build_byte_map
      byte_map = (@@unicode_to_byte ||= build_byte_map)
      inv = Hash(UInt8, Char).new
      byte_map.each { |ch, b| inv[b] = ch }
      inv
    end

    private def self.build_byte_map : Hash(Char, UInt8)
      # Printable byte ranges that map to themselves
      bs = [] of Int32
      (('!'.ord)..('~'.ord)).each { |b| bs << b }
      (('¡'.ord)..('¬'.ord)).each { |b| bs << b }
      (('®'.ord)..('ÿ'.ord)).each { |b| bs << b }
      cs = bs.dup
      n = 0
      (0..255).each do |b|
        unless bs.includes?(b)
          bs << b
          cs << (256 + n)
          n += 1
        end
      end
      map = Hash(Char, UInt8).new
      bs.each_with_index { |b, i| map[cs[i].chr] = b.to_u8 }
      map
    end

    def initialize
      @vocab = Hash(String, Int32).new
      @inv_vocab = [] of String
      @merges = [] of Tuple(String, String)
      @merges_map = Hash(Tuple(String, String), String).new
      @merges_rank = Hash(Tuple(String, String), Int32).new
    end

    # Load a pre-trained tokenizer from a HuggingFace tokenizer.json file
    def self.from_hf(path : String) : BPETokenizer
      tok = new
      tok.hf_mode = true
      data = JSON.parse(File.read(path))
      model = data["model"]

      # Load vocab
      model["vocab"].as_h.each do |token, id|
        tok.vocab[token] = id.as_i
      end

      # Build inverse vocab
      max_id = tok.vocab.values.max? || 0
      tok.inv_vocab.concat(Array(String).new(max_id + 1, ""))
      tok.vocab.each { |token, id| tok.inv_vocab[id] = token }

      # Load merges with rank (handles both "a b" string and ["a","b"] array formats)
      if merges = model["merges"]?
        merges.as_a.each_with_index do |m, rank|
          parts = if m.as_a?
                    a = m.as_a
                    next unless a.size == 2
                    [a[0].as_s, a[1].as_s]
                  else
                    m.as_s.split(' ', 2)
                  end
          next unless parts.size == 2
          pair = {parts[0], parts[1]}
          merged = parts[0] + parts[1]
          tok.merges << pair
          tok.merges_map[pair] = merged
          tok.merges_rank[pair] = rank
        end
      end

      # Load added tokens (special tokens)
      if added = data["added_tokens"]?
        added.as_a.each do |t|
          token = t["content"].as_s
          id = t["id"].as_i
          tok.vocab[token] = id
          while tok.inv_vocab.size <= id
            tok.inv_vocab << ""
          end
          tok.inv_vocab[id] = token
        end
      end

      tok
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
        # Use CPU-only pair counting - faster and simpler than GPU version
        pair_counts = cpu_pair_counts(corpus, freqs)
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
      if @hf_mode
        return encode_hf(text)
      end
      words = text.split(/\s+/)
      ids = Array(Int32).new(words.size)
      words.each do |word|
        tokens = encode_tokens(word)
        tokens.each { |t| ids << add_token(t) }
      end
      ids
    end

    # HF-style encode: split on spaces, prefix with Ġ, apply BPE merges
    private def encode_hf(text : String) : Array(Int32)
      ids = Array(Int32).new
      return ids if text.empty?
      # Convert raw bytes to GPT-2 unicode representation
      unicode_text = String.build do |s|
        text.bytes.each { |b| s << BPETokenizer.byte_to_unicode(b) }
      end
      # Split on spaces: each space becomes Ġ prefix on the following word
      # Use regex to split into chunks: spaces attach to the following word
      parts = unicode_text.scan(/\S+|\s+/)
      parts.each do |m|
        word = m[0]
        tokens = encode_tokens_hf(word)
        tokens.each do |t|
          if id = @vocab[t]?
            ids << id
          end
        end
      end
      ids
    end

    private def encode_tokens_hf(word : String) : Array(String)
      tokens = word.chars.map(&.to_s)
      return tokens if @merges_rank.empty? || tokens.size <= 1

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
      if @hf_mode
        # Byte-level BPE: each token char maps back to a raw byte (GPT-2
        # bytes_to_unicode reverse), then the byte stream is UTF-8 decoded.
        bytes = Array(UInt8).new
        ids.each do |id|
          token = @inv_vocab[id]? || ""
          token.each_char do |ch|
            bytes << BPETokenizer.unicode_to_byte(ch)
          end
        end
        return String.new(Slice.new(bytes.to_unsafe, bytes.size))
      end
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
      loop do
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
