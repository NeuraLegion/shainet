module SHAInet
  # Simple embedding lookup table. Maps integer token IDs to vectors of floats.
  class EmbeddingLayer < Layer
    property embeddings : Hash(Int32, Array(Float64))
    property gradients : Hash(Int32, Array(Float64))
    getter current_ids : Array(Int32)

    def initialize(l_size : Int32, activation_function : ActivationFunction = SHAInet.none)
      super("memory", l_size, activation_function)
      @embeddings = Hash(Int32, Array(Float64)).new
      @gradients = Hash(Int32, Array(Float64)).new
      @current_ids = [] of Int32
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
      @current_ids << id
      vec
    end

    # Accumulate gradient for the last embedded ids
    def accumulate_gradient
      until @current_ids.empty?
        id = @current_ids.shift
        grad = @gradients[id] ||= Array(Float64).new(@l_size, 0.0)
        @neurons.each_with_index do |n, i|
          grad[i] += n.gradient
        end
      end
    end

    # Update embeddings using stored gradients and clear them
    def apply_gradients(lr : Float64)
      @gradients.each do |id, grad|
        emb = lookup(id)
        grad.each_with_index do |g, i|
          emb[i] -= lr*g
          grad[i] = 0.0
        end
      end
    end
  end
end
