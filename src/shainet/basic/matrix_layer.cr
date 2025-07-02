require "../math/simple_matrix"
require "../math/cuda_matrix"
require "../math/unified_matrix"

module SHAInet
  class MatrixLayer
    # Base layer properties
    property :n_type
    getter :activation_function, :l_size
    property weights : SimpleMatrix | CudaMatrix
    property biases : SimpleMatrix | CudaMatrix
    property g_w : SimpleMatrix | CudaMatrix
    property g_b : SimpleMatrix | CudaMatrix
    getter size : Int32

    # Stored forward pass data for backpropagation
    @input : SimpleMatrix | CudaMatrix | Nil
    @activations : SimpleMatrix | CudaMatrix | Nil
    @sigma_primes : SimpleMatrix | CudaMatrix | Nil

    def initialize(in_size : Int32, @size : Int32)
      @n_type = "memory"
      @l_size = @size
      @activation_function = SHAInet.sigmoid
      # Use SimpleMatrix for consistency during debugging
      mat_klass = SimpleMatrix
      @weights = mat_klass.new(in_size, @size).random_fill!
      @biases = mat_klass.new(1, @size).random_fill!
      @g_w = mat_klass.zeros(in_size, @size)
      @g_b = mat_klass.zeros(1, @size)
      @input = nil
      @activations = nil
      @sigma_primes = nil
    end

    # Constructor for compatibility with Layer API
    def initialize(@n_type : String, @size : Int32, @activation_function : ActivationFunction = SHAInet.sigmoid)
      @l_size = @size
      # Use SimpleMatrix for consistency during debugging
      mat_klass = SimpleMatrix
      @weights = mat_klass.new(1, @size).random_fill!
      @biases = mat_klass.new(1, @size).random_fill!
      @g_w = mat_klass.zeros(1, @size)
      @g_b = mat_klass.zeros(1, @size)
      @input = nil
      @activations = nil
      @sigma_primes = nil
    end

    # Constructor with custom activation function
    def initialize(in_size : Int32, @size : Int32, @activation_function : ActivationFunction)
      @n_type = "memory"
      @l_size = @size
      # Use SimpleMatrix for consistency during debugging
      mat_klass = SimpleMatrix
      @weights = mat_klass.new(in_size, @size).random_fill!
      @biases = mat_klass.new(1, @size).random_fill!
      @g_w = mat_klass.zeros(in_size, @size)
      @g_b = mat_klass.zeros(1, @size)
      @input = nil
      @activations = nil
      @sigma_primes = nil
    end

    def inspect
      Log.info { @n_type }
    end

    # Forward pass - the main method that should be used
    def forward(input : SimpleMatrix | CudaMatrix) : SimpleMatrix | CudaMatrix
      @input = input
      mat_klass = input.class

      # Ensure weights and biases are compatible with input type
      w = mat_klass == @weights.class ? @weights : mat_klass.from_a(@weights.to_a)
      b = mat_klass == @biases.class ? @biases : mat_klass.from_a(@biases.to_a)

      # Linear transformation: input * weights + bias
      linear_result = input * w
      linear_result.add_bias!(b)

      # Apply activation function and store derivatives for backprop
      @activations = linear_result.clone
      @sigma_primes = mat_klass.new(linear_result.rows, linear_result.cols)

      linear_result.rows.times do |i|
        linear_result.cols.times do |j|
          val = linear_result[i, j]
          activation_val, derivative_val = @activation_function.call(val)
          @activations.not_nil![i, j] = activation_val
          @sigma_primes.not_nil![i, j] = derivative_val
        end
      end

      @activations.not_nil!
    end

    # Backward pass - accumulates gradients and returns gradient for previous layer
    def backward(grad : SimpleMatrix | CudaMatrix) : SimpleMatrix | CudaMatrix
      return grad if @input.nil? || @sigma_primes.nil?

      input = @input.not_nil!
      sigma_primes = @sigma_primes.not_nil!

      # Apply activation derivative: grad ⊙ σ'
      local_grad = grad.clone
      local_grad.rows.times do |i|
        local_grad.cols.times do |j|
          local_grad[i, j] = grad[i, j] * sigma_primes[i, j]
        end
      end

      # Accumulate weight gradients: ∂L/∂W += input^T * local_grad
      @g_w = @g_w + input.transpose * local_grad

      # Accumulate bias gradients: ∂L/∂b += sum(local_grad, axis=0)
      local_grad.rows.times do |i|
        local_grad.cols.times do |j|
          @g_b[0, j] += local_grad[i, j]
        end
      end

      # Return gradient for previous layer: local_grad * W^T
      w = local_grad.class == @weights.class ? @weights : local_grad.class.from_a(@weights.to_a)
      local_grad * w.transpose
    end

    # Update weights using accumulated gradients
    def update_weights(learning_rate : Float64)
      # W := W - lr * ∂L/∂W
      # b := b - lr * ∂L/∂b
      @weights = @weights - @g_w * learning_rate
      @biases = @biases - @g_b * learning_rate
    end

    # Reset gradients to zero
    def zero_gradients
      @g_w.rows.times do |i|
        @g_w.cols.times do |j|
          @g_w[i, j] = 0.0
        end
      end
      @g_b.rows.times do |i|
        @g_b.cols.times do |j|
          @g_b[i, j] = 0.0
        end
      end
    end

    # Getter for activations (for testing and debugging)
    def activations
      @activations.not_nil!
    end
  end
end
