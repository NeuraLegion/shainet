require "log"

module SHAInet
  class Filter
    getter input_surface : Array(Int32), window_size : Int32, stride : Int32, padding : Int32, activation_function : ActivationFunction
    property weights : SimpleMatrix
    property biases : SimpleMatrix
    property activations : SimpleMatrix
    property gradients : SimpleMatrix
    property output : Array(Float64)

    # Add property for weight gradients
    property weight_gradients : SimpleMatrix
    property bias_gradient : Float64

    def initialize(@input_surface : Array(Int32), # expecting [width, height, channels]
                   @padding : Int32 = 0,
                   @window_size : Int32 = 1,
                   @stride : Int32 = 1,
                   @activation_function : ActivationFunction = SHAInet.none)

      # Calculate weight dimensions based on input surface and window size
      output_width = @input_surface[0]
      output_height = @input_surface[1]
      channels = @input_surface[2]

      # Initialize weights, biases, activations, and gradients
      @weights = SimpleMatrix.new(@window_size * @window_size * channels, 1)
      @weights.random_fill!(-0.1, 0.1)  # Use standard method without bang

      @biases = SimpleMatrix.new(1, 1)
      @biases.random_fill!(-0.1, 0.1)

      @activations = SimpleMatrix.new(output_height, output_width)
      @gradients = SimpleMatrix.new(output_height, output_width)

      # Initialize output array
      @output = Array(Float64).new(output_width * output_height) { 0.0 }

      # Initialize weight gradients
      @weight_gradients = SimpleMatrix.new(@window_size * @window_size * channels, 1)
      @bias_gradient = 0.0
    end

    # Forward pass using matrix operations - full convolution implementation
    def forward(input_matrix : SimpleMatrix) : SimpleMatrix
      # Parse dimensions
      output_width = @input_surface[0]
      output_height = @input_surface[1]
      channels = @input_surface[2]

      # Reshape input for convolution
      input_width = Math.sqrt(input_matrix.cols / channels).to_i
      input_height = input_width  # Assuming square input

      # Initialize result matrix
      result = SimpleMatrix.new(output_height, output_width)

      # Perform convolution by sliding the window over the input
      output_height.times do |out_y|
        output_width.times do |out_x|
          # Initialize with bias
          val = @biases[0, 0]

          # Calculate top-left corner of the window in the input
          in_y_start = out_y * @stride - @padding
          in_x_start = out_x * @stride - @padding

          # Convolve over the window
          weight_idx = 0
          channels.times do |c|
            # Calculate channel offset in input
            channel_offset = c * input_width * input_height

            @window_size.times do |wy|
              @window_size.times do |wx|
                # Calculate position in input
                in_y = in_y_start + wy
                in_x = in_x_start + wx

                # Check if position is valid (not in padding)
                if in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width
                  # Get input value and corresponding weight
                  input_idx = channel_offset + in_y * input_width + in_x
                  if input_idx < input_matrix.cols
                    input_val = input_matrix[0, input_idx]
                    val += input_val * @weights[weight_idx, 0]
                  end
                end

                weight_idx += 1
              end
            end
          end

          # Apply activation function
          act, _ = @activation_function.call(val)
          result[out_y, out_x] = act
          @output[out_y * output_width + out_x] = act
        end
      end

      # Store activations for backward pass
      @activations = result.clone

      result
    end

    # Backward pass - proper convolution backpropagation
    def backward(input_matrix : SimpleMatrix, gradient_matrix : SimpleMatrix) : SimpleMatrix
      # Store gradients for this filter
      @gradients = gradient_matrix.clone

      # Parse dimensions
      output_width = @input_surface[0]
      output_height = @input_surface[1]
      channels = @input_surface[2]

      # Calculate input dimensions
      input_width = Math.sqrt(input_matrix.cols / channels).to_i
      input_height = input_width  # Assuming square input

      # Initialize gradients for previous layer
      prev_gradients = SimpleMatrix.new(1, input_matrix.cols)

      # Initialize weight gradients
      weight_gradients = SimpleMatrix.new(@weights.rows, @weights.cols)
      bias_gradient = 0.0

      # Process each output gradient
      output_height.times do |out_y|
        output_width.times do |out_x|
          # Get gradient for this output position
          out_gradient = @gradients[out_y, out_x]

          # Add to bias gradient
          bias_gradient += out_gradient

          # Calculate input window position
          in_y_start = out_y * @stride - @padding
          in_x_start = out_x * @stride - @padding

          # Calculate gradients for weights and inputs
          weight_idx = 0
          channels.times do |c|
            # Calculate channel offset in input
            channel_offset = c * input_width * input_height

            @window_size.times do |wy|
              @window_size.times do |wx|
                # Calculate position in input
                in_y = in_y_start + wy
                in_x = in_x_start + wx

                # Check if position is valid (not in padding)
                if in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width
                  # Get input index
                  input_idx = channel_offset + in_y * input_width + in_x

                  if input_idx < input_matrix.cols
                    # Update weight gradient
                    input_val = input_matrix[0, input_idx]
                    weight_gradients[weight_idx, 0] += input_val * out_gradient

                    # Update input gradient
                    prev_gradients[0, input_idx] += @weights[weight_idx, 0] * out_gradient
                  end
                end

                weight_idx += 1
              end
            end
          end
        end
      end

      # Store weight gradients for weight update
      @weight_gradients = weight_gradients
      @bias_gradient = bias_gradient

      prev_gradients
    end

    # Update weights based on gradients
    def update_weights(learning_rate : Float64)
      # Apply weight updates using calculated gradients
      @weights.rows.times do |i|
        @weights.cols.times do |j|
          @weights[i, j] -= learning_rate * @weight_gradients[i, j]
        end
      end

      # Update bias
      @biases[0, 0] -= learning_rate * @bias_gradient

      # Reset gradients for next iteration
      @weight_gradients.rows.times do |i|
        @weight_gradients.cols.times do |j|
          @weight_gradients[i, j] = 0.0
        end
      end
      @bias_gradient = 0.0
    end
  end
end
