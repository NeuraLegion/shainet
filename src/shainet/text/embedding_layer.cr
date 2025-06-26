module SHAInet
  # Simple embedding lookup table. Maps integer token IDs to vectors of floats.
  class EmbeddingLayer < Layer
    property embeddings : SimpleMatrix
    property gradients : SimpleMatrix
    getter current_ids : Array(Int32)

    def initialize(vocab_size : Int32, l_size : Int32, activation_function : ActivationFunction = SHAInet.none)
      super("memory", l_size, activation_function)
      mat_klass = CUDA.available? ? CudaMatrix : SimpleMatrix
      @embeddings = mat_klass.new(vocab_size, l_size).random_fill!
      @gradients = mat_klass.zeros(vocab_size, l_size)
      @current_ids = [] of Int32
    end

    # Migration helper for legacy models using hash based storage
    def self.from_hash(hash : Hash(Int32, Array(Float64)), activation_function : ActivationFunction = SHAInet.none)
      vocab_size = hash.keys.max? ? hash.keys.max + 1 : 0
      l_size = hash.values.first?.try(&.size) || 0
      layer = new(vocab_size, l_size, activation_function)
      hash.each do |id, vals|
        vals.each_with_index { |v, i| layer.embeddings[id, i] = v }
      end
      layer
    end

    # Retrieve embedding vector for the given token id. If the token id does not
    # exist in the table, it is initialized with random values.
    def lookup(id : Int32) : Array(Float64)
      Array.new(@l_size) { |i| @embeddings[id, i] }
    end

    # Set the neuron activations for this layer according to the embedding of the
    # provided token id. Returns the embedding vector.
    def embed(id : Int32) : Array(Float64)
      @neurons.each_with_index do |n, i|
        n.activation = @embeddings[id, i]
      end
      @current_ids << id
      lookup(id)
    end

    # Accumulate gradient for the last embedded ids
    def accumulate_gradient
      until @current_ids.empty?
        id = @current_ids.shift
        @neurons.each_with_index do |n, i|
          @gradients[id, i] += n.gradient
        end
      end
    end

    # Update embeddings using stored gradients and clear them
    def apply_gradients(lr : Float64)
      @gradients.rows.times do |r|
        @gradients.cols.times do |c|
          g = @gradients[r, c]
          next if g == 0.0
          @embeddings[r, c] -= lr * g
          @gradients[r, c] = 0.0
        end
      end
    end
  end
end
