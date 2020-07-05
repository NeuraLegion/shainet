require "apatite"

module SHAInet
  class Layer
    include Apatite
    Log = ::Log.for(self)

    property :n_type, :neurons
    getter :activation_function, :l_size
    property input_sums : Matrix(Float64), weights : Matrix(Float64), biases : Matrix(Float64)
    getter activations : Matrix(Float64), sigma_primes : Matrix(Float64)

    def initialize(@n_type : String, @l_size : Int32, @activation_function : ActivationFunction = SHAInet.sigmoid)
      @neurons = Array(Neuron).new

      # ------- Experimental -------
      # Pointer matrices for forward propogation
      @input_sums = Matrix(Float64).build(1, @l_size) { 0.0 }
      @weights = Matrix(Float64).build(1, @l_size) { 0.0 }
      @biases = Matrix(Float64).build(1, @l_size) { 0.0 }
      @activations = Matrix(Float64).build(1, @l_size) { 0.0 }
      @sigma_primes = Matrix(Float64).build(1, @l_size) { 0.0 }

      # # Pointer matrices for back propogation
      # @w_gradients = Array(Array(Pointer)).new
      # @b_gradients = Array(Pointer).new
      # @prev_weights = Array(Array(Pointer)).new
      # @prev_biases = Array(Pointer).new

      # Populate layer with neurons and save pointers
      @l_size.times do |i|
        neuron = Neuron.new(@n_type)

        @neurons << neuron

        # ------- Experimental -------
        @input_sums[0, i] = neuron.input_sum
        @biases[0, i] = neuron.bias
        @activations[0, i] = neuron.activation
        @sigma_primes[0, i] = neuron.sigma_prime

        # @prev_bias[0, i] = neuron.prev_bias_ptr
        # @b_gradients[0, i] = neuron.gradient_ptr
      end

      # ------- Experimental -------
      # Transpose the needed matrices
      @input_sums.t
      @biases.t
      @activations.t
      @sigma_primes.t
    end

    def clone
      layer_old = self
      layer_new = Layer.new(layer_old.n_type, layer_old.@l_size, layer_old.activation_function)

      layer_new.neurons = layer_old.neurons.clone
      layer_new
    end

    # If you don't want neurons to have a blank memory of builds
    def random_seed
      @neurons.each do |neuron|
        neuron.activation = rand(-1_f64..1_f64)
      end
      Log.info { "Layers seeded with random values" }
    end

    # If you want to change the type of layer including all neuron types within it
    def type_change(new_neuron_type : String)
      raise NeuralNetRunError.new("Must define correct neuron type, if you're not sure choose \"memory\" as a default") if NEURON_TYPES.any? { |x| x == new_neuron_type } == false
      @neurons.each { |neuron| neuron.n_type = new_neuron_type }
      Log.info { "Layer type changed from #{@n_type} to #{new_neuron_type}" }
      @n_type = new_neuron_type
    end

    def inspect
      Log.info { @n_type }
      Log.info { @neurons }
    end

    def size
      @l_size
    end
  end
end
