require "json"

module SHAInet
  class Layer
    include JSON::Serializable

    property :n_type, :neurons
    getter :activation_function, :l_size
    getter input_sums : PtrMatrix, weights : PtrMatrix, biases : PtrMatrix
    getter activations : PtrMatrix, sigma_primes : PtrMatrix

    @[JSON::Field(ignore: true)]
    @input_sums : PtrMatrix = PtrMatrix.new(width: @l_size, height: 1)

    @[JSON::Field(ignore: true)]
    @activations : PtrMatrix = PtrMatrix.new(width: @l_size, height: 1)

    @[JSON::Field(ignore: true)]
    @weights : PtrMatrix = PtrMatrix.new(width: 1, height: 1) # temp matrix

    @[JSON::Field(ignore: true)]
    @sigma_primes : PtrMatrix = PtrMatrix.new(width: @l_size, height: 1)

    @[JSON::Field(ignore: true)]
    @biases : PtrMatrix = PtrMatrix.new(width: @l_size, height: 1)

    @[JSON::Field(ignore: true)]
    @activation_function : ActivationFunction = SHAInet.sigmoid

    @[JSON::Field(ignore: true)]
    @logger : Logger = Logger.new(STDOUT)

    @neurons : Array(SHAInet::Neuron) = Array(SHAInet::Neuron).new

    def initialize(@n_type : String, @l_size : Int32, @activation_function : ActivationFunction = SHAInet.sigmoid, @logger : Logger = Logger.new(STDOUT))
      after_initialize
    end

    protected def after_initialize
      # Populate layer with neurons and save pointers
      @l_size.times do |i|
        neuron = Neuron.new(@n_type)

        @neurons << neuron
        # ------- Experimental -------
        @input_sums.data[0][i] = neuron.input_sum_ptr
        @biases.data[0][i] = neuron.bias_ptr
        @activations.data[0][i] = neuron.activation_ptr
        @sigma_primes.data[0][i] = neuron.sigma_prime_ptr

        # @prev_bias.data[0][i] = neuron.prev_bias_ptr
        # @b_gradients.data[0][i] = neuron.gradient_ptr
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
      return layer_new
    end

    # If you don't want neurons to have a blank memory of zeros
    def random_seed
      @neurons.each do |neuron|
        neuron.activation = rand(-1_f64..1_f64)
      end
      @logger.info("Layers seeded with random values")
    end

    # If you want to change the type of layer including all neuron types within it
    def type_change(new_neuron_type : String)
      raise NeuralNetRunError.new("Must define correct neuron type, if you're not sure choose \"memory\" as a default") if NEURON_TYPES.any? { |x| x == new_neuron_type } == false
      @neurons.each { |neuron| neuron.n_type = new_neuron_type }
      @logger.info("Layer type chaged from #{@n_type} to #{new_neuron_type}")
      @n_type = new_neuron_type
    end

    def inspect
      @logger.info(@n_type)
      @logger.info(@neurons)
    end

    def size
      return @l_size
    end
  end
end
