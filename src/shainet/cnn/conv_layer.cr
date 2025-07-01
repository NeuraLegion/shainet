require "log"

module SHAInet
  class ConvLayer
    Log = ::Log.for(self)

    getter master_network : CNN, prev_layer : CNNLayer | ConvLayer, filters : Array(Filter)
    getter window_size : Int32, stride : Int32, padding : Int32, activation_function : ActivationFunction

    # Matrix-based properties
    property input_matrix : SimpleMatrix
    property output_matrix : SimpleMatrix
    property gradients : SimpleMatrix

    def initialize(@master_network : CNN,
                   @prev_layer : ConvLayer | CNNLayer,
                   filters_num : Int32 = 1,
                   @window_size : Int32 = 1,
                   @stride : Int32 = 1,
                   @padding : Int32 = 0,
                   @activation_function : ActivationFunction = SHAInet.none)
      #
      raise CNNInitializationError.new("ConvLayer must have at least one filter") if filters_num < 1
      raise CNNInitializationError.new("Padding value must be Int32 >= 0") if @padding < 0
      raise CNNInitializationError.new("Window size value must be Int32 >= 1") if @window_size < 1
      raise CNNInitializationError.new("Stride value must be Int32 >= 1") if @stride < 1

      # Get dimensions from previous layer
      width = 0
      height = 0
      channels = 0

      if @prev_layer.is_a?(InputLayer)
        width = @prev_layer.as(InputLayer).input_volume[0]
        height = @prev_layer.as(InputLayer).input_volume[1]
        channels = @prev_layer.as(InputLayer).input_volume[2]
      else
        width = @prev_layer.filters.first.input_surface[0]
        height = @prev_layer.filters.first.input_surface[1]
        channels = @prev_layer.filters.size
      end

      # This is a calculation to make sure the input volume matches a correct desired output volume
      output_width = ((width - @window_size + 2*@padding)/@stride + 1)
      unless output_width.to_i == output_width
        raise CNNInitializationError.new("Output volume must be a whole number, change: window size, stride and/or padding")
      end

      # Create filters with appropriate dimensions
      @filters = Array(Filter).new(filters_num) do
        Filter.new([output_width.to_i, output_width.to_i, channels], @padding, @window_size, @stride, @activation_function)
      end

      # Initialize matrix properties
      @input_matrix = SimpleMatrix.new(1, width * height * channels)
      @output_matrix = SimpleMatrix.new(1, filters_num * output_width.to_i * output_width.to_i)
      @gradients = SimpleMatrix.new(1, 1) # Will be properly sized during backprop
    end

    # Activate method using matrix operations
    def activate
      # Get input from previous layer as a matrix
      prepare_input_matrix

      # Process through each filter and collect outputs
      output_width = @filters.first.input_surface[0]
      output_height = @filters.first.input_surface[1]

      # Apply each filter
      @filters.each_with_index do |filter, filter_idx|
        filter_output = filter.forward(@input_matrix)

        # Copy filter output to the output matrix
        output_width.times do |y|
          output_height.times do |x|
            @output_matrix[0, filter_idx * output_width * output_height + y * output_width + x] =
              filter_output[y, x]
          end
        end
      end
    end

    # Helper to prepare input matrix from previous layer
    private def prepare_input_matrix
      if @prev_layer.is_a?(InputLayer)
        # For input layer, the data is already provided as a matrix
        @input_matrix = @prev_layer.as(InputLayer).output_matrix.clone
      else
        # For other layers, we need to flatten the filter outputs
        width = @prev_layer.filters.first.input_surface[0]
        height = @prev_layer.filters.first.input_surface[1]
        channels = @prev_layer.filters.size

        # Initialize or resize input matrix if needed
        if @input_matrix.rows != 1 || @input_matrix.cols != width * height * channels
          @input_matrix = SimpleMatrix.new(1, width * height * channels)
        end

        # Copy data from previous layer filters to input matrix
        @prev_layer.filters.each_with_index do |filter, c|
          # Assuming filter.output contains the flattened output
          filter.output.each_with_index do |val, i|
            @input_matrix[0, c * width * height + i] = val
          end
        end
      end
    end

    # Backward pass for error propagation
    def error_prop(batch : Bool = false)
      # Initialize gradients for the previous layer
      prev_gradients = prepare_prev_gradients

      # Process each filter's backward pass
      @filters.each_with_index do |filter, filter_idx|
        # Extract gradients for this filter from the overall gradients
        filter_gradients = extract_filter_gradients(filter_idx)

        # Compute gradients for the filter and add to previous layer gradients
        filter_prev_gradients = filter.backward(@input_matrix, filter_gradients)

        # Accumulate gradients for previous layer
        prev_gradients.rows.times do |r|
          prev_gradients.cols.times do |c|
            prev_gradients[r, c] += filter_prev_gradients[r, c]
          end
        end
      end

      # Propagate gradients to previous layer if it supports matrix-based backprop
      if @prev_layer.responds_to?(:backward_matrix)
        @prev_layer.backward_matrix(self, prev_gradients)
      end
    end

    # Helper to prepare gradient matrix for previous layer
    private def prepare_prev_gradients
      if @prev_layer.is_a?(InputLayer)
        width = @prev_layer.as(InputLayer).input_volume[0]
        height = @prev_layer.as(InputLayer).input_volume[1]
        channels = @prev_layer.as(InputLayer).input_volume[2]
      else
        width = @prev_layer.filters.first.input_surface[0]
        height = @prev_layer.filters.first.input_surface[1]
        channels = @prev_layer.filters.size
      end

      # Create gradient matrix matching the input shape
      SimpleMatrix.new(1, width * height * channels)
    end

    # Helper to extract gradients for a specific filter
    private def extract_filter_gradients(filter_idx)
      output_width = @filters.first.input_surface[0]
      output_height = @filters.first.input_surface[1]

      # Create gradient matrix for this filter
      filter_gradients = SimpleMatrix.new(output_height, output_width)

      # Copy gradients for this filter from the overall gradients
      output_height.times do |y|
        output_width.times do |x|
          # Assuming @gradients contains the gradients for all filters
          filter_gradients[y, x] =
            @gradients[0, filter_idx * output_width * output_height + y * output_width + x]
        end
      end

      filter_gradients
    end

    # Update weights and biases for all filters
    def update_wb(learn_type : Symbol | String, batch : Bool = false)
      # For simplified implementation, we'll just use a fixed learning rate
      learning_rate = 0.01

      @filters.each do |filter|
        filter.update_weights(learning_rate)
      end
    end

    # For debugging
    def inspect(what : String)
      puts "##################################################"
      puts "ConvLayer with #{@filters.size} filters:"
      puts "Window size: #{@window_size}, Stride: #{@stride}, Padding: #{@padding}"
      puts "----------"

      case what
      when "weights"
        @filters.each_with_index do |filter, i|
          puts "Filter #{i} weights:"
          filter.weights.rows.times do |r|
            row_vals = (0...filter.weights.cols).map { |c| filter.weights[r, c].round(4) }
            puts "Row #{r}: #{row_vals}"
          end
        end
      when "bias"
        @filters.each_with_index do |filter, i|
          puts "Filter #{i} bias: #{filter.biases[0, 0].round(4)}"
        end
      when "activations"
        @filters.each_with_index do |filter, i|
          puts "Filter #{i} activations:"
          filter.activations.rows.times do |r|
            row_vals = (0...filter.activations.cols).map { |c| filter.activations[r, c].round(4) }
            puts "Row #{r}: #{row_vals}"
          end
        end
      when "gradients"
        @filters.each_with_index do |filter, i|
          puts "Filter #{i} gradients:"
          filter.gradients.rows.times do |r|
            row_vals = (0...filter.gradients.cols).map { |c| filter.gradients[r, c].round(4) }
            puts "Row #{r}: #{row_vals}"
          end
        end
      end
      puts "------------------------------------------------"
    end
  end
end
