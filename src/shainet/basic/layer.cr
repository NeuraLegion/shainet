require "../math/simple_matrix"

module SHAInet
  class Layer
    Log = ::Log.for(self)

    property :n_type, :neurons
    getter :activation_function, :l_size
    property input_sums : SimpleMatrix, weights : SimpleMatrix, biases : SimpleMatrix
    getter activations : SimpleMatrix, sigma_primes : SimpleMatrix

    def initialize(@n_type : String, @l_size : Int32, @activation_function : ActivationFunction = SHAInet.sigmoid)
      @neurons = Array(Neuron).new

      # ------- Experimental -------
      # Pointer matrices for forward propogation
      @input_sums = SimpleMatrix.new(1, @l_size, 0.0)
      @weights = SimpleMatrix.new(1, @l_size, 0.0)
      @biases = SimpleMatrix.new(1, @l_size, 0.0)
      @activations = SimpleMatrix.new(1, @l_size, 0.0)
      @sigma_primes = SimpleMatrix.new(1, @l_size, 0.0)

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
      @input_sums.transpose
      @biases.transpose
      @activations.transpose
      @sigma_primes.transpose
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

    # Forward propagation using matrix multiplication. Returns the resulting
    # activation matrix and updates neuron states for compatibility with the
    # non-matrix implementation.
    def forward_matrix(input : SimpleMatrix | CudaMatrix)
      mat_klass = input.class
      w = if mat_klass == CudaMatrix && @weights.is_a?(CudaMatrix)
            @weights
          elsif mat_klass == SimpleMatrix && @weights.is_a?(SimpleMatrix)
            @weights
          else
            mat_klass.from_a(@weights.to_a)
          end
      b = if mat_klass == CudaMatrix && @biases.is_a?(CudaMatrix)
            @biases
          elsif mat_klass == SimpleMatrix && @biases.is_a?(SimpleMatrix)
            @biases
          else
            mat_klass.from_a(@biases.to_a)
          end

      out = input * w.transpose
      out.add_bias!(b)

      out.rows.times do |i|
        out.cols.times do |j|
          val = out[i, j]
          act, sig = @activation_function.call(val)
          out[i, j] = act
          if i == out.rows - 1
            neuron = @neurons[j]
            neuron.activation = act
            neuron.sigma_prime = sig
            neuron.input_sum = val
          end
        end
      end
      out
    end

    # Backward propagation using matrix multiplication. Calculates the gradient
    # for this layer based on the next layer's weights and gradients. The
    # returned matrix can be used to continue propagating errors backwards.
    def backward_matrix(next_layer : Layer, next_grad : SimpleMatrix | CudaMatrix? = nil)
      mat_klass = next_grad ? next_grad.class : (CUDA.available? ? CudaMatrix : SimpleMatrix)
      grad = next_grad || mat_klass.from_a([next_layer.neurons.map(&.gradient)])
      w = if mat_klass == CudaMatrix && next_layer.weights.is_a?(CudaMatrix)
            next_layer.weights
          elsif mat_klass == SimpleMatrix && next_layer.weights.is_a?(SimpleMatrix)
            next_layer.weights
          else
            mat_klass.from_a(next_layer.weights.to_a)
          end

      err = grad * w
      err.rows.times do |i|
        err.cols.times do |j|
          val = err[i, j] * @neurons[j].sigma_prime
          err[i, j] = val
          if i == err.rows - 1
            @neurons[j].gradient = val
          end
        end
      end
      err
    end
  end
end
