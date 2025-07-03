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
      # Use CudaMatrix when CUDA is fully available, otherwise SimpleMatrix
      mat_klass = CUDA.fully_available? ? CudaMatrix : SimpleMatrix
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
      # Use CudaMatrix when CUDA is fully available, otherwise SimpleMatrix
      mat_klass = CUDA.fully_available? ? CudaMatrix : SimpleMatrix
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
      # Use CudaMatrix when CUDA is fully available, otherwise SimpleMatrix
      mat_klass = CUDA.fully_available? ? CudaMatrix : SimpleMatrix
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

      # Assume weights and biases already match the input device type
      # No conversion needed - they should have been allocated correctly
      w = @weights
      b = @biases

      # Linear transformation: input * weights + bias
      linear_result = input * w
      linear_result.add_bias!(b)

      # Apply activation function and store derivatives for backprop
      @activations = linear_result.clone
      @sigma_primes = mat_klass.new(linear_result.rows, linear_result.cols)

      # Use GPU kernels for activation when CUDA is fully available
      if CUDA.fully_available? && linear_result.is_a?(CudaMatrix) && @activation_function == SHAInet.sigmoid
        activations_cuda = @activations.as(CudaMatrix)
        sigma_primes_cuda = @sigma_primes.as(CudaMatrix)
        linear_cuda = linear_result.as(CudaMatrix)

        # Ensure all matrices are synced to GPU
        linear_cuda.sync_to_device! unless linear_cuda.device_dirty?

        size = linear_result.rows * linear_result.cols
        CUDA.sigmoid_forward(
          activations_cuda.device_ptr.not_nil!,
          sigma_primes_cuda.device_ptr.not_nil!,
          linear_cuda.device_ptr.not_nil!,
          size
        )

        # Mark results as dirty on device
        activations_cuda.mark_device_dirty!
        sigma_primes_cuda.mark_device_dirty!
      else
        # CPU fallback for non-sigmoid or non-CUDA cases
        linear_result.rows.times do |i|
          linear_result.cols.times do |j|
            val = linear_result[i, j]
            activation_val, derivative_val = @activation_function.call(val)
            @activations.not_nil![i, j] = activation_val
            @sigma_primes.not_nil![i, j] = derivative_val
          end
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

      # Use GPU kernel for gradient computation when CUDA is fully available
      if CUDA.fully_available? && local_grad.is_a?(CudaMatrix) && sigma_primes.is_a?(CudaMatrix)
        local_grad_cuda = local_grad.as(CudaMatrix)
        grad_cuda = grad.as(CudaMatrix)
        sigma_primes_cuda = sigma_primes.as(CudaMatrix)

        # Ensure matrices are synced to GPU
        grad_cuda.sync_to_device! unless grad_cuda.device_dirty?
        sigma_primes_cuda.sync_to_device! unless sigma_primes_cuda.device_dirty?

        size = local_grad.rows * local_grad.cols
        CUDA.apply_gradient(
          local_grad_cuda.device_ptr.not_nil!,
          grad_cuda.device_ptr.not_nil!,
          sigma_primes_cuda.device_ptr.not_nil!,
          size
        )

        local_grad_cuda.mark_device_dirty!
      else
        # CPU fallback
        local_grad.rows.times do |i|
          local_grad.cols.times do |j|
            local_grad[i, j] = grad[i, j] * sigma_primes[i, j]
          end
        end
      end

      # Accumulate weight gradients: ∂L/∂W += input^T * local_grad
      @g_w = @g_w + input.transpose * local_grad

      # Accumulate bias gradients: ∂L/∂b += sum(local_grad, axis=0)
      if CUDA.fully_available? && local_grad.is_a?(CudaMatrix) && @g_b.is_a?(CudaMatrix)
        local_grad_cuda = local_grad.as(CudaMatrix)
        g_b_cuda = @g_b.as(CudaMatrix)

        # Ensure matrices are synced to GPU
        local_grad_cuda.sync_to_device! unless local_grad_cuda.device_dirty?
        g_b_cuda.sync_to_device! unless g_b_cuda.device_dirty?

        CUDA.accumulate_bias_grad(
          g_b_cuda.device_ptr.not_nil!,
          local_grad_cuda.device_ptr.not_nil!,
          local_grad.rows,
          local_grad.cols
        )

        g_b_cuda.mark_device_dirty!
      else
        # CPU fallback
        local_grad.rows.times do |i|
          local_grad.cols.times do |j|
            @g_b[0, j] += local_grad[i, j]
          end
        end
      end

      # Return gradient for previous layer: local_grad * W^T
      # No conversion needed - weights should already match the input device type
      local_grad * @weights.transpose
    end

    # Update weights using accumulated gradients
    def update_weights(learning_rate : Float64)
      # W := W - lr * ∂L/∂W
      # b := b - lr * ∂L/∂b
      @weights = @weights - @g_w * learning_rate
      @biases = @biases - @g_b * learning_rate

      # Mark CudaMatrix weights/biases as dirty after update
      if @weights.is_a?(CudaMatrix)
        @weights.as(CudaMatrix).mark_device_dirty!
      end
      if @biases.is_a?(CudaMatrix)
        @biases.as(CudaMatrix).mark_device_dirty!
      end
    end

    # Reset gradients to zero
    def zero_gradients
      # Use GPU kernel for zeroing when CUDA is fully available
      if CUDA.fully_available? && @g_w.is_a?(CudaMatrix) && @g_b.is_a?(CudaMatrix)
        g_w_cuda = @g_w.as(CudaMatrix)
        g_b_cuda = @g_b.as(CudaMatrix)

        # Zero weight gradients
        w_size = @g_w.rows * @g_w.cols
        CUDA.zero_matrix(g_w_cuda.device_ptr.not_nil!, w_size)
        g_w_cuda.mark_device_dirty!

        # Zero bias gradients
        b_size = @g_b.rows * @g_b.cols
        CUDA.zero_matrix(g_b_cuda.device_ptr.not_nil!, b_size)
        g_b_cuda.mark_device_dirty!
      else
        # CPU fallback
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
    end

    # Getter for activations (for testing and debugging)
    def activations
      @activations.not_nil!
    end
  end
end
