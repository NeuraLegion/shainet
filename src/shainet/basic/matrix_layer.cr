require "../math/simple_matrix"
require "../math/cuda_matrix"

module SHAInet
  class MatrixLayer
    # Base layer properties
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
    def initialize(@size : Int32, @activation_function : ActivationFunction = SHAInet.sigmoid)
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

    # Convert layer matrices to GPU
    def to_gpu!
      if CUDA.fully_available?
        @weights = @weights.as(SimpleMatrix).to_cuda unless @weights.is_a?(CudaMatrix)
        @biases = @biases.as(SimpleMatrix).to_cuda unless @biases.is_a?(CudaMatrix)
        @g_w = @g_w.as(SimpleMatrix).to_cuda unless @g_w.is_a?(CudaMatrix)
        @g_b = @g_b.as(SimpleMatrix).to_cuda unless @g_b.is_a?(CudaMatrix)
        @input = @input.as(SimpleMatrix).to_cuda if @input && @input.is_a?(SimpleMatrix)
        @activations = @activations.as(SimpleMatrix).to_cuda if @activations && @activations.is_a?(SimpleMatrix)
        @sigma_primes = @sigma_primes.as(SimpleMatrix).to_cuda if @sigma_primes && @sigma_primes.is_a?(SimpleMatrix)
      end
    end

    # Convert layer matrices to CPU
    def to_cpu!
      if @weights.is_a?(CudaMatrix)
        @weights = @weights.as(CudaMatrix).to_simple
        @biases = @biases.as(CudaMatrix).to_simple
        @g_w = @g_w.as(CudaMatrix).to_simple
        @g_b = @g_b.as(CudaMatrix).to_simple
        @input = @input.as(CudaMatrix).to_simple if @input && @input.is_a?(CudaMatrix)
        @activations = @activations.as(CudaMatrix).to_simple if @activations && @activations.is_a?(CudaMatrix)
        @sigma_primes = @sigma_primes.as(CudaMatrix).to_simple if @sigma_primes && @sigma_primes.is_a?(CudaMatrix)
      end
    end

    # GPU path - all CudaMatrix operations
    def forward(input : CudaMatrix) : CudaMatrix
      @input = input

      # Weights and biases should already be CudaMatrix in GPU path
      w = @weights.as(CudaMatrix)
      b = @biases.as(CudaMatrix)

      # Linear transformation: input * weights + bias
      linear_result = input * w
      linear_result.add_bias!(b)

      # Apply activation function and store derivatives for backprop
      @activations = linear_result.clone
      @sigma_primes = CudaMatrix.new(linear_result.rows, linear_result.cols)

      activations_cuda = @activations.as(CudaMatrix)
      sigma_primes_cuda = @sigma_primes.as(CudaMatrix)

      # Ensure all matrices are synced to GPU
      linear_result.sync_to_device!("matrix_layer_forward") unless linear_result.device_dirty?

      size = linear_result.rows * linear_result.cols

      case @activation_function
      when SHAInet.sigmoid
        CUDA.sigmoid_forward(
          activations_cuda.device_ptr.not_nil!,
          sigma_primes_cuda.device_ptr.not_nil!,
          linear_result.device_ptr.not_nil!,
          size
        )
        # Mark results as dirty on device
        activations_cuda.mark_device_dirty!
        sigma_primes_cuda.mark_device_dirty!
      when SHAInet.none
        # Identity function: input passes through unchanged, derivatives are all 1.0
        @activations = linear_result
        @sigma_primes = CudaMatrix.ones(linear_result.rows, linear_result.cols)
        @sigma_primes.as(CudaMatrix).mark_device_dirty!
      else
        # For other activation functions, fall back to CPU
        linear_result.rows.times do |i|
          linear_result.cols.times do |j|
            val = linear_result[i, j]
            activation_val, derivative_val = @activation_function.call(val)
            @activations.not_nil![i, j] = activation_val
            @sigma_primes.not_nil![i, j] = derivative_val
          end
        end
      end

      @activations.as(CudaMatrix)
    end

    # CPU path - all SimpleMatrix operations
    def forward(input : SimpleMatrix) : SimpleMatrix
      @input = input

      # Weights and biases should already be SimpleMatrix in CPU path
      w = @weights.as(SimpleMatrix)
      b = @biases.as(SimpleMatrix)

      # Linear transformation: input * weights + bias
      linear_result = input * w
      linear_result.add_bias!(b)

      # Apply activation function and store derivatives for backprop
      @activations = linear_result.clone
      @sigma_primes = SimpleMatrix.new(linear_result.rows, linear_result.cols)

      # CPU activation computation
      linear_result.rows.times do |i|
        linear_result.cols.times do |j|
          val = linear_result[i, j]
          activation_val, derivative_val = @activation_function.call(val)
          @activations.not_nil![i, j] = activation_val
          @sigma_primes.not_nil![i, j] = derivative_val
        end
      end

      @activations.as(SimpleMatrix)
    end

    # GPU path backward - all CudaMatrix operations
    def backward(grad : CudaMatrix) : CudaMatrix
      return grad if @input.nil? || @sigma_primes.nil?

      input = @input.as(CudaMatrix)
      sigma_primes = @sigma_primes.as(CudaMatrix)

      # Apply activation derivative: grad ⊙ σ'
      local_grad = grad.clone

      # Use GPU kernel for gradient computation
      grad.sync_to_device!("matrix_layer_backward") unless grad.device_dirty?
      sigma_primes.sync_to_device!("matrix_layer_backward") unless sigma_primes.device_dirty?

      size = local_grad.rows * local_grad.cols
      CUDA.apply_gradient(
        local_grad.device_ptr.not_nil!,
        grad.device_ptr.not_nil!,
        sigma_primes.device_ptr.not_nil!,
        size
      )

      local_grad.mark_device_dirty!

      # Accumulate weight gradients: ∂L/∂W += input^T * local_grad
      @g_w = @g_w.as(CudaMatrix) + input.transpose * local_grad

      # Accumulate bias gradients: ∂L/∂b += sum(local_grad, axis=0)
      g_b_cuda = @g_b.as(CudaMatrix)
      local_grad.sync_to_device!("matrix_layer_bias_grad") unless local_grad.device_dirty?
      g_b_cuda.sync_to_device!("matrix_layer_bias_grad") unless g_b_cuda.device_dirty?

      CUDA.accumulate_bias_grad(
        g_b_cuda.device_ptr.not_nil!,
        local_grad.device_ptr.not_nil!,
        local_grad.rows,
        local_grad.cols
      )

      g_b_cuda.mark_device_dirty!

      # Return gradient for previous layer: local_grad * W^T
      local_grad * @weights.as(CudaMatrix).transpose
    end

    # CPU path backward - all SimpleMatrix operations
    def backward(grad : SimpleMatrix) : SimpleMatrix
      return grad if @input.nil? || @sigma_primes.nil?

      input = @input.as(SimpleMatrix)
      sigma_primes = @sigma_primes.as(SimpleMatrix)

      # Apply activation derivative: grad ⊙ σ'
      local_grad = grad.clone

      # CPU gradient computation
      local_grad.rows.times do |i|
        local_grad.cols.times do |j|
          local_grad[i, j] = grad[i, j] * sigma_primes[i, j]
        end
      end

      # Accumulate weight gradients: ∂L/∂W += input^T * local_grad
      @g_w = @g_w.as(SimpleMatrix) + input.transpose * local_grad

      # Accumulate bias gradients: ∂L/∂b += sum(local_grad, axis=0)
      local_grad.rows.times do |i|
        local_grad.cols.times do |j|
          @g_b[0, j] += local_grad[i, j]
        end
      end

      # Return gradient for previous layer: local_grad * W^T
      local_grad * @weights.as(SimpleMatrix).transpose
    end

    # Update weights using accumulated gradients - device-specific versions
    def update_weights(learning_rate : Float64)
      # Check device type and call appropriate method
      if @weights.is_a?(CudaMatrix)
        update_weights_gpu(learning_rate)
      else
        update_weights_cpu(learning_rate)
      end
    end

    # GPU path weight update - all CudaMatrix operations
    private def update_weights_gpu(learning_rate : Float64)
      # W := W - lr * ∂L/∂W
      # b := b - lr * ∂L/∂b
      @weights = @weights.as(CudaMatrix) - @g_w.as(CudaMatrix) * learning_rate
      @biases = @biases.as(CudaMatrix) - @g_b.as(CudaMatrix) * learning_rate

      # Mark CudaMatrix weights/biases as dirty after update
      @weights.as(CudaMatrix).mark_device_dirty!
      @biases.as(CudaMatrix).mark_device_dirty!
    end

    # CPU path weight update - all SimpleMatrix operations
    private def update_weights_cpu(learning_rate : Float64)
      # W := W - lr * ∂L/∂W
      # b := b - lr * ∂L/∂b
      @weights = @weights.as(SimpleMatrix) - @g_w.as(SimpleMatrix) * learning_rate
      @biases = @biases.as(SimpleMatrix) - @g_b.as(SimpleMatrix) * learning_rate
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
        # CPU fallback - create new zero matrices
        @g_w = SimpleMatrix.zeros(@g_w.rows, @g_w.cols)
        @g_b = SimpleMatrix.zeros(@g_b.rows, @g_b.cols)
      end
    end

    # Getter for activations (for testing and debugging)
    def activations
      @activations.not_nil!
    end
  end
end
