module SHAInet
  # Very small tokenizer used for toy examples. It builds a vocabulary of words
  # from given text and encodes/decodes sentences to arrays of token IDs.
  class Tokenizer
    getter vocab : Hash(String, Int32)
    getter inv_vocab : Array(String)

    def initialize
      @vocab = Hash(String, Int32).new
      @inv_vocab = [] of String
    end

    # Update the vocabulary with all unique words from the given text. Splits the
    # text on whitespace.
    def build(text : String)
      text.split(/\s+/).each do |token|
        add_token(token)
      end
    end

    # Convert a string into an array of token IDs. Unknown tokens are added to
    # the vocabulary.
    def encode(text : String) : Array(Int32)
      text.split(/\s+/).map do |token|
        add_token(token)
      end
    end

    # Convert an array of token IDs back to their corresponding words. Unknown
    # IDs are returned as an empty string.
    def decode(ids : Array(Int32)) : Array(String)
      ids.map { |id| @inv_vocab[id]? || "" }
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
