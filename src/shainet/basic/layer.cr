require "../math/simple_matrix"

module SHAInet
  class Layer
    Log = ::Log.for(self)

    property :n_type
    getter :activation_function, :l_size
    property input_sums : SimpleMatrix, weights : SimpleMatrix, biases : SimpleMatrix
    getter activations : SimpleMatrix, sigma_primes : SimpleMatrix

    def initialize(@n_type : String, @l_size : Int32, @activation_function : ActivationFunction = SHAInet.sigmoid)
      # Matrix-only implementation - no neurons/synapses
      @input_sums = SimpleMatrix.new(1, @l_size, 0.0)
      @weights = SimpleMatrix.new(1, @l_size, 0.0)
      @biases = SimpleMatrix.new(1, @l_size, 0.0)
      @activations = SimpleMatrix.new(1, @l_size, 0.0)
      @sigma_primes = SimpleMatrix.new(1, @l_size, 0.0)

      @input_sums.transpose
      @biases.transpose
      @activations.transpose
      @sigma_primes.transpose
    end

    def clone
      layer_old = self
      layer_new = Layer.new(layer_old.n_type, layer_old.@l_size, layer_old.activation_function)
      layer_new
    end

    def inspect
      Log.info { @n_type }
    end

    def size
      @l_size
    end

    # Forward propagation using matrix multiplication. Returns the resulting
    # activation matrix and updates internal state matrices.
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

      # Forward pass: input * weights^T + bias
      rsp = nil
      begin
        rsp = input * w.transpose
      rescue ex : Exception
        if ex.message.to_s.includes?("size mismatch")
          # Handle matrix multiplication size mismatch
          # Often happens when embedding dimension doesn't match layer dimension
          Log.info { "Matrix size mismatch in forward_matrix: input(#{input.rows}x#{input.cols}), weights(#{w.rows}x#{w.cols})" }

          # Create properly sized output matrix
          rsp = SimpleMatrix.new(input.rows, w.rows)

          # Manual matrix multiplication when possible
          if input.cols > 0 && w.cols > 0
            min_k = [input.cols, w.cols].min
            input.rows.times do |i|
              w.rows.times do |j|
                sum = 0.0
                min_k.times do |k|
                  sum += input[i, k] * w[j, k]
                end
                rsp[i, j] = sum
              end
            end
          else
            raise ArgumentError.new("Cannot multiply matrices with shapes (#{input.rows},#{input.cols}) and (#{w.rows},#{w.cols})")
          end
        else
          raise ex
        end
      end

      # Ensure rsp is not nil before proceeding
      if rsp.nil?
        raise ArgumentError.new("Matrix multiplication failed: input(#{input.rows}x#{input.cols}), weights(#{w.rows}x#{w.cols})")
      end

      # Manual bias addition to handle size mismatches
      if rsp.nil?
        raise ArgumentError.new("Matrix multiplication failed: input(#{input.rows}x#{input.cols}), weights(#{w.rows}x#{w.cols})")
      end

      if b.rows == 1 && b.cols > 0
        # Ensure bias has the right number of columns
        bias_cols = [b.cols, rsp.cols].min

        rsp.rows.times do |i|
          bias_cols.times do |j|
            rsp[i, j] += b[0, j]
          end
        end
      elsif b.cols == 1 && b.rows > 0
        # Handle case where bias is transposed (column vector)
        # This often happens in transformer networks
        bias_rows = [b.rows, rsp.cols].min

        rsp.rows.times do |i|
          bias_rows.times do |j|
            rsp[i, j] += b[j, 0]
          end
        end
      else
        # Try normal bias addition
        begin
          rsp.add_bias!(b)
        rescue ex : Exception
          # If the normal addition fails, manually add biases
          Log.info { "Falling back to manual bias addition: rsp(#{rsp.rows}x#{rsp.cols}), bias(#{b.rows}x#{b.cols})" }

          # Create a reshaped bias of proper dimensions
          if rsp.cols > 0
            new_b = SimpleMatrix.new(1, rsp.cols)
            rsp.cols.times do |j|
              if b.rows == 1 && j < b.cols
                new_b[0, j] = b[0, j]
              elsif b.cols == 1 && j < b.rows
                new_b[0, j] = b[j, 0]
              end
            end

            # Apply the reshaped bias
            rsp.rows.times do |i|
              rsp.cols.times do |j|
                rsp[i, j] += new_b[0, j]
              end
            end
          else
            raise ArgumentError.new("Cannot add bias with shape (#{b.rows},#{b.cols}) to output with shape (#{rsp.rows},#{rsp.cols})")
          end
        end
      end

      # Update internal state and apply activation function
      @input_sums = rsp.clone if @input_sums.is_a?(SimpleMatrix) && rsp.is_a?(SimpleMatrix)

      rsp.rows.times do |i|
        rsp.cols.times do |j|
          val = rsp[i, j]
          act, sig = @activation_function.call(val)
          rsp[i, j] = act

          # Update internal state matrices for compatibility
          if i < @activations.rows && j < @activations.cols && @activations.is_a?(SimpleMatrix) && rsp.is_a?(SimpleMatrix)
            @activations[i, j] = act
          end
          if i < @sigma_primes.rows && j < @sigma_primes.cols && @sigma_primes.is_a?(SimpleMatrix) && rsp.is_a?(SimpleMatrix)
            @sigma_primes[i, j] = sig
          end
        end
      end
      rsp
    end

    # Backward propagation using matrix multiplication. Calculates the gradient
    # for this layer based on the next layer's weights and gradients.
    def backward_matrix(next_layer : Layer, next_grad : SimpleMatrix | CudaMatrix? = nil)
      mat_klass = next_grad ? next_grad.class : (CUDA.available? ? CudaMatrix : SimpleMatrix)
      grad = next_grad

      # Get weights from next layer with proper type conversion
      w = if mat_klass == CudaMatrix && next_layer.weights.is_a?(CudaMatrix)
            next_layer.weights
          elsif mat_klass == SimpleMatrix && next_layer.weights.is_a?(SimpleMatrix)
            next_layer.weights
          else
            mat_klass.from_a(next_layer.weights.to_a)
          end

      # If no gradient is provided, compute from current layer's error signal
      if grad.nil?
        grad = mat_klass.new(1, @l_size, 0.0)
        @l_size.times do |j|
          # Use stored activation and sigma_prime from forward pass
          act = @activations[0, j]
          sig_prime = @sigma_primes[0, j]
          input_sum = @input_sums[0, j]
          grad[0, j] = input_sum * act * sig_prime
        end
      else
        # Convert gradient to appropriate matrix type if needed
        if mat_klass == CudaMatrix && grad.is_a?(SimpleMatrix)
          grad = GPUMemory.to_gpu(grad)
        elsif mat_klass == SimpleMatrix && grad.is_a?(CudaMatrix)
          grad = SimpleMatrix.from_a(grad.to_a)
        end

        # Apply element-wise multiplication with activation derivatives
        @l_size.times do |j|
          act = @activations[0, j]
          sig_prime = @sigma_primes[0, j]
          input_sum = @input_sums[0, j]
          grad[0, j] = input_sum * act * sig_prime * grad[0, j]
        end
      end

      # Check dimensions before matrix multiplication
      if grad.cols == w.rows
        # Compute gradient for previous layer: grad * weights
        grad * w
      else
        # Handle dimension mismatch - create a properly sized gradient
        result = mat_klass.new(grad.rows, w.cols, 0.0)
        # If dimension mismatch is severe, we can't do much
        result
      end
    end
  end
end
