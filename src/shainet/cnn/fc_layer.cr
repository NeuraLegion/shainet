require "log"

module SHAInet
  class FullyConnectedLayer
    Log = ::Log.for(self)

    getter filters : Array(Filter), prev_layer : CNNLayer | ConvLayer
    getter output : Array(Float64)
    getter weights : SimpleMatrix
    getter biases : SimpleMatrix
    getter activations : SimpleMatrix
    getter sigma_primes : SimpleMatrix

    def initialize(@master_network : CNN,
                   @prev_layer : CNNLayer | ConvLayer,
                   l_size : Int32,
                   @activation_function : ActivationFunction = SHAInet.none)
      #
      # since this is similar to a classic layer, we store all neurons in a single array
      filters = height = 1
      width = l_size

      @filters = Array(Filter).new(filters) { Filter.new([width, height, filters]) }

      @output = Array(Float64).new(l_size) { 0.0 }

      # Matrix-based properties (new)
      input_size = get_input_size
      @weights = SimpleMatrix.new(l_size, input_size)
      @weights.random_fill!

      @biases = SimpleMatrix.new(l_size, 1)
      @biases.random_fill!

      # For storing activations and derivatives
      @activations = SimpleMatrix.new(1, l_size) # Row vector for output
      @sigma_primes = SimpleMatrix.new(1, l_size) # Row vector for derivatives
    end

    private def get_input_size
      if @prev_layer.is_a?(ConvLayer)
        total_inputs = 0
        # Use safe access to prev_layer filters
        @prev_layer.as(ConvLayer).filters.each do |filter|
          # Approximate the input size from filter dimensions
          filter_width = filter.input_surface[0]
          filter_height = filter.input_surface[1]
          total_inputs += filter_width * filter_height
        end
        total_inputs
      else
        @prev_layer.output.size
      end
    end

    def activate
      # Get input from previous layer
      input_vector = if @prev_layer.is_a?(ConvLayer)
                      # Flatten all filter outputs into a single vector
                      inputs = [] of Float64
                      @prev_layer.as(ConvLayer).filters.each do |filter|
                        filter.output.each do |val|
                          inputs << val
                        end
                      end
                      inputs
                    else
                      @prev_layer.output
                    end

      # Convert to matrix
      input_matrix = SimpleMatrix.from_a([input_vector])

      # Forward pass: input * weights^T + bias
      z = input_matrix * @weights.transpose

      # Add bias
      l_size = @output.size
      l_size.times do |i|
        z[0, i] += @biases[i, 0]
      end

      # Apply activation function and store results
      l_size.times do |i|
        act, sigma_prime = @activation_function.call(z[0, i])
        @activations[0, i] = act
        @sigma_primes[0, i] = sigma_prime
        @output[i] = act
      end
    end

    def error_prop(batch : Bool = false)
      # Simplified to just call backward_matrix if available
      if @prev_layer.responds_to?(:backward_matrix)
        @prev_layer.backward_matrix(self, @activations)
      end
    end

    def update_weights(learning_rate : Float64)
      # Update weights based on gradients
      @weights.rows.times do |i|
        @weights.cols.times do |j|
          @weights[i, j] -= learning_rate * @activations[0, i]
        end
      end

      # Update biases
      @biases.rows.times do |i|
        @biases[i, 0] -= learning_rate * @activations[0, i]
      end
    end

    def forward_matrix(input_matrix : SimpleMatrix) : SimpleMatrix
      # Process batch inputs
      batch_size = input_matrix.rows

      # Create output matrix for batch
      output = SimpleMatrix.new(batch_size, @output.size)

      # Process each example in the batch
      batch_size.times do |b|
        # Extract this example's input
        example_input = SimpleMatrix.new(1, input_matrix.cols)
        input_matrix.cols.times do |c|
          example_input[0, c] = input_matrix[b, c]
        end

        # Forward pass
        z = example_input * @weights.transpose

        # Add bias and apply activation
        z.cols.times do |i|
          z[0, i] += @biases[i, 0]
          act, sigma_prime = @activation_function.call(z[0, i])
          output[b, i] = act

          # Update internal state for last example (for backward pass)
          if b == z.rows - 1
            @activations[0, i] = act
            @sigma_primes[0, i] = sigma_prime
          end
        end
      end

      output
    end

    def backward_matrix(prev_layer, gradient_matrix : SimpleMatrix) : SimpleMatrix
      # Calculate the derivative of the loss with respect to pre-activation values
      deltas = SimpleMatrix.new(1, @output.size)

      # Apply derivative of activation function to gradient
      @output.size.times do |i|
        deltas[0, i] = gradient_matrix[0, i] * @sigma_primes[0, i]
      end

      # Get input from previous layer
      input_vector = if prev_layer.is_a?(ConvLayer)
                      # Flatten all filter outputs into a single vector
                      inputs = [] of Float64
                      prev_layer.as(ConvLayer).filters.each do |filter|
                        filter.output.each do |val|
                          inputs << val
                        end
                      end
                      inputs
                    else
                      prev_layer.output
                    end

      # Convert to matrix
      input_matrix = SimpleMatrix.from_a([input_vector])

      # Calculate weight gradients: deltas^T * input
      weight_gradients = deltas.transpose * input_matrix

      # Calculate input gradients: deltas * weights
      input_gradients = deltas * @weights

      # Update weights
      learning_rate = 0.01  # This should ideally be passed in
      @weights.rows.times do |i|
        @weights.cols.times do |j|
          @weights[i, j] -= learning_rate * weight_gradients[i, j]
        end
      end

      # Update biases
      @biases.rows.times do |i|
        @biases[i, 0] -= learning_rate * deltas[0, i]
      end

      input_gradients
    end

    def inspect(what : String)
      puts "##################################################"
      puts "FullyConnectedLayer:"
      puts "----------"
      case what
      when "weights"
        puts "Matrix weights:"
        @weights.rows.times do |r|
          row_vals = (0...@weights.cols).map { |c| @weights[r, c].round(4) }
          puts "Row #{r}: #{row_vals}"
        end
      when "bias"
        puts "Matrix biases:"
        @biases.rows.times do |r|
          puts "Row #{r}: #{@biases[r, 0].round(4)}"
        end
      when "activations"
        puts "Matrix activations:"
        @activations.rows.times do |r|
          row_vals = (0...@activations.cols).map { |c| @activations[r, c].round(4) }
          puts "Row #{r}: #{row_vals}"
        end
      when "gradients"
        puts "No explicit gradients matrix"
      end
      puts "------------------------------------------------"
    end
  end
end
