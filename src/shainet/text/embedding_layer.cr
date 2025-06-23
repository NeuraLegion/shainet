module SHAInet
  # Simple embedding lookup table. Maps integer token IDs to vectors of floats.
  class EmbeddingLayer < Layer
    property embeddings : Hash(Int32, Array(Float64))

    def initialize(l_size : Int32, activation_function : ActivationFunction = SHAInet.none)
      super("memory", l_size, activation_function)
      @embeddings = Hash(Int32, Array(Float64)).new
    end

    # Retrieve embedding vector for the given token id. If the token id does not
    # exist in the table, it is initialized with random values.
    def lookup(id : Int32) : Array(Float64)
      @embeddings[id] ||= Array(Float64).new(@l_size) { rand(-0.1_f64..0.1_f64) }
    end

    # Set the neuron activations for this layer according to the embedding of the
    # provided token id. Returns the embedding vector.
    def embed(id : Int32) : Array(Float64)
      vec = lookup(id)
      vec.each_with_index do |v, i|
        @neurons[i].activation = v
      end
      vec
    end
  end
end
