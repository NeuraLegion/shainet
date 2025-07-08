require "log"
require "json"
{% if flag?(:enable_cuda) %}
  require "../cuda"
  require "../cudnn"
{% else %}
  require "../cuda_stub"
{% end %}
require "../math/simple_matrix"
require "../math/cuda_matrix"

module SHAInet
  class Network
    # ------------
    # This is a matrix-based neural network implementation. All operations
    # are performed on matrices rather than individual neurons/synapses.
    # This approach provides better performance and GPU acceleration capabilities.

    # This file contains all the methods for running and training the network,
    # for methods regarding creating and maintaining go to network_setup.cr
    # ------------

    @batch_in_ws : CudaMatrix? = nil
    @batch_out_ws : CudaMatrix? = nil
    @batch_grad_ws : CudaMatrix? = nil

    # Run an input through the network to get an output (weights & biases do not change)
    # Simple wrapper that converts array input to matrix and calls the core matrix method
    def run(input : Array(GenNum), stealth : Bool = false) : Array(Float64)
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      raise NeuralNetRunError.new(
        "Error input data size: #{input.size} doesn't fit input layer size: #{expected_size}.") unless input.size == expected_size

      # Convert to matrix and use core matrix method
      processed = @mixed_precision ? input.map { |v| v.to_f32.to_f64 } : input.map(&.to_f64)
      matrix = GPUMemory.to_gpu(SimpleMatrix.from_a([processed]))
      result_matrix = run(matrix, stealth: stealth)

      # Efficient array extraction - only sync once if needed
      output = if result_matrix.is_a?(CudaMatrix)
                 result_matrix.sync_from_device!("array_output") if result_matrix.device_dirty?
                 result_matrix.to_flat_array
               else
                 result_matrix.to_a.first
               end

      unless stealth
        Log.info { "Input => #{input}, network output => #{output}" }
      end
      output
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # Overload allowing retrieval of the raw matrix
    def run(input : Array(GenNum), *, return_matrix : Bool, stealth : Bool = false) : Array(Float64) | CudaMatrix | SimpleMatrix
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      raise NeuralNetRunError.new(
        "Error input data size: #{input.size} doesn't fit input layer size: #{expected_size}.") unless input.size == expected_size

      processed = @mixed_precision ? input.map { |v| v.to_f32.to_f64 } : input.map(&.to_f64)
      matrix = GPUMemory.to_gpu(SimpleMatrix.from_a([processed]))
      result_matrix = run(matrix, stealth: stealth)

      if return_matrix
        result_matrix
      else
        output = if result_matrix.is_a?(CudaMatrix)
                   result_matrix.sync_from_device!("array_output") if result_matrix.device_dirty?
                   result_matrix.to_flat_array
                 else
                   result_matrix.to_a.first
                 end

        Log.info { "Input => #{input}, network output => #{output}" } unless stealth
        output
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # GPU path - all CudaMatrix operations
    def run(input : CudaMatrix, stealth : Bool = false) : CudaMatrix
      verify_net_before_train

      matrix = input

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            # Ensure embedding layer is on GPU for GPU path
            l.as(EmbeddingLayer).to_gpu!
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = extract_tokens_gpu(matrix)
            matrix = l.as(EmbeddingLayer).embed(tokens)
          when TransformerLayer
            # Ensure transformer layer is on GPU for GPU path
            l.as(TransformerLayer).to_gpu!
            matrix = l.as(TransformerLayer).forward(matrix)
          else
            # Handle MatrixLayer and other layer types
            l.to_gpu! if l.responds_to?(:to_gpu!)
            matrix = l.forward(matrix)
          end
        end
        out_layer = @output_layers.last
        # Ensure output layer is on GPU for GPU path
        out_layer.to_gpu!
        w = out_layer.weights.as(CudaMatrix)
        b = out_layer.biases.as(CudaMatrix)
        matrix = safe_output_transform(matrix.as(CudaMatrix), w)
        matrix.add_bias!(b)

        # Apply activation function - use GPU kernels when available
        unless out_layer.activation_function == SHAInet.identity
          # Try to use GPU kernels for common activation functions
          if try_gpu_activation(matrix, out_layer.activation_function)
            # GPU activation succeeded, update internal state matrices if needed
            if out_layer.responds_to?(:activations) && out_layer.responds_to?(:sigma_primes)
              # For now, keep last row handling simple - this is a rare case
              last_row_vals = matrix.rows == 1 ? matrix : slice_rows_helper(matrix, matrix.rows - 1, 1)
              out_layer.activations = last_row_vals.clone
              out_layer.sigma_primes = CudaMatrix.ones(1, matrix.cols)
            end
          else
            # Fallback to CPU for unsupported activation functions - minimize sync operations
            matrix.sync_from_device!("activation_fallback")
            # Use unsafe_get/unsafe_set for better performance
            matrix.rows.times do |i|
              matrix.cols.times do |j|
                val = matrix.unsafe_get(i, j)
                act, sig = out_layer.activation_function.call(val)
                matrix.unsafe_set(i, j, act)
                if i == matrix.rows - 1
                  # Update internal state matrices for output layer
                  if out_layer.responds_to?(:activations) && out_layer.responds_to?(:sigma_primes)
                    out_layer.activations.unsafe_set(0, j, act) if out_layer.activations.is_a?(CudaMatrix)
                    out_layer.sigma_primes.unsafe_set(0, j, sig) if out_layer.sigma_primes.is_a?(CudaMatrix)
                  end
                end
              end
            end
            matrix.sync_to_device!("activation_fallback")
          end
        end
        matrix.as(CudaMatrix)
      else
        # Standard matrix processing for non-transformer networks
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            # Ensure embedding layer is on GPU for GPU path
            l.as(EmbeddingLayer).to_gpu!
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = extract_tokens_gpu(matrix)
            matrix = l.as(EmbeddingLayer).embed(tokens)
          else
            # Handle MatrixLayer and other layer types
            l.to_gpu! if l.responds_to?(:to_gpu!)
            matrix = l.forward(matrix)
          end
        end
        out_layer = @output_layers.last
        # Ensure output layer is on GPU for GPU path
        out_layer.to_gpu!
        matrix = out_layer.forward(matrix)
        matrix.as(CudaMatrix)
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # CPU path - all SimpleMatrix operations
    def run(input : SimpleMatrix, stealth : Bool = false) : SimpleMatrix
      verify_net_before_train

      matrix = input

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = (0...matrix.rows).map { |r| matrix[r, 0].to_i }
            matrix = l.as(EmbeddingLayer).embed_cpu(tokens)
          when TransformerLayer
            matrix = l.as(TransformerLayer).forward(matrix)
          end
        end
        out_layer = @output_layers.last
        w = out_layer.weights.as(SimpleMatrix)
        b = out_layer.biases.as(SimpleMatrix)
        matrix = safe_output_transform(matrix.as(SimpleMatrix), w)

        # CPU bias addition
        matrix.rows.times do |i|
          matrix.cols.times do |j|
            matrix[i, j] += b[0, j]
          end
        end

        # Apply activation function - for identity, no operation needed
        unless out_layer.activation_function == SHAInet.identity
          matrix.rows.times do |i|
            matrix.cols.times do |j|
              val = matrix[i, j]
              act, sig = out_layer.activation_function.call(val)
              matrix[i, j] = act
              if i == matrix.rows - 1
                # Update internal state matrices for output layer
                if out_layer.responds_to?(:activations) && out_layer.responds_to?(:sigma_primes)
                  out_layer.activations[0, j] = act
                  out_layer.sigma_primes[0, j] = sig
                end
              end
            end
          end
        end
        matrix.as(SimpleMatrix)
      else
        # Standard matrix processing for non-transformer networks
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = (0...matrix.rows).map { |r| matrix[r, 0].to_i }
            matrix = l.as(EmbeddingLayer).embed(tokens)
          else
            matrix = l.forward(matrix)
          end
        end
        out_layer = @output_layers.last
        matrix = out_layer.forward(matrix)
        matrix.as(SimpleMatrix)
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # Run a batch of sequences by calling `run` for each sequence
    # This is a convenience wrapper that can be consolidated with more direct matrix operations
    def run(input : Array(Array(Array(GenNum))), stealth : Bool = false) : Array(Array(Array(Float64)))
      input.map { |seq| run(seq, stealth: stealth) }
    end

    def run(input : Array(Array(Array(GenNum))), *, return_matrix : Bool, stealth : Bool = false) : Array(Array(Array(Float64))) | Array(CudaMatrix | SimpleMatrix)
      if return_matrix
        input.map { |seq| run(seq, stealth: stealth, return_matrix: true) }
      else
        input.map { |seq| run(seq, stealth: stealth) }
      end
    end

    # Accept a sequence of integer tokens for embedding layers
    # This is a convenience wrapper around the standard run method
    def run(input : Array(Array(Int32)), stealth : Bool = false) : Array(Array(Float64))
      seq = input.map { |x| x.map(&.to_f64) }
      run(seq, stealth: stealth)
    end

    def run(input : Array(Array(Int32)), *, return_matrix : Bool, stealth : Bool = false) : Array(Array(Float64)) | CudaMatrix | SimpleMatrix
      seq = input.map { |x| x.map(&.to_f64) }
      run(seq, stealth: stealth, return_matrix: return_matrix)
    end

    # Accept integer input for embedding layers
    # This is a convenience wrapper around the standard run method
    def run(input : Array(Int32), stealth : Bool = false) : Array(Float64)
      float_in = input.map(&.to_f64)
      run(float_in, stealth: stealth)
    end

    def run(input : Array(Int32), *, return_matrix : Bool, stealth : Bool = false) : Array(Float64) | CudaMatrix | SimpleMatrix
      float_in = input.map(&.to_f64)
      run(float_in, stealth: stealth, return_matrix: return_matrix)
    end

    # Accept sequence input - converts to matrix and calls core matrix method
    def run(input : Array(Array(GenNum)), stealth : Bool = false) : Array(Array(Float64))
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      input.each do |step|
        raise NeuralNetRunError.new("Error input data size: #{step.size} doesn't fit input layer size: #{expected_size}.") unless step.size == expected_size
      end

      # Convert to matrix and use core matrix method
      processed = input.map do |x|
        @mixed_precision ? x.map { |v| v.to_f32.to_f64 } : x.map(&.to_f64)
      end
      matrix = GPUMemory.to_gpu(SimpleMatrix.from_a(processed))
      result_matrix = run(matrix, stealth: stealth)

      # Efficient array extraction - sync only once if needed
      if result_matrix.is_a?(CudaMatrix)
        result_matrix.sync_from_device!("run_output") if result_matrix.device_dirty?
        result_matrix.to_a
      else
        result_matrix.to_a
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    def run(input : Array(Array(GenNum)), *, return_matrix : Bool, stealth : Bool = false) : Array(Array(Float64)) | CudaMatrix | SimpleMatrix
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      input.each do |step|
        raise NeuralNetRunError.new("Error input data size: #{step.size} doesn't fit input layer size: #{expected_size}.") unless step.size == expected_size
      end

      processed = input.map do |x|
        @mixed_precision ? x.map { |v| v.to_f32.to_f64 } : x.map(&.to_f64)
      end
      matrix = GPUMemory.to_gpu(SimpleMatrix.from_a(processed))
      result_matrix = run(matrix, stealth: stealth)

      if return_matrix
        result_matrix
      else
        if result_matrix.is_a?(CudaMatrix)
          result_matrix.sync_from_device!("run_output") if result_matrix.device_dirty?
          result_matrix.to_a
        else
          result_matrix.to_a
        end
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # Quantifies how good the network performed for a single input compared to the expected output
    # This function returns the actual output and updates the error gradient for the output layer
    def evaluate(input_data : Array(GenNum),
                 expected_output : Array(GenNum),
                 cost_function : CostFunction = SHAInet.quadratic_cost)
      processed = @mixed_precision ? input_data.map { |v| v.to_f32.to_f64 } : input_data.map(&.to_f64)
      actual_output = run(processed, stealth: true)

      # Test for NaNs & exploading gradients
      validate_values(actual_output, "actual_output")

      # Get the error signal for the final layer, based on the cost function
      @error_signal = [] of Float64 # Collect all the errors for current run

      actual_output.size.times do |i|
        cost = cost_function.call(expected_output[i], actual_output[i])
        @error_signal << cost[:value]

        # puts "Actual output: #{actual_output}"
        # puts "Cost value: #{cost[:value]}"
        # puts "Cost derivative: #{cost[:derivative]}"
        # puts "---"
      end

      # Test for NaNs & exploading gradients
      validate_values(@error_signal, "error_signal")
      @total_error = @error_signal.reduce(0.0) { |acc, i| acc + i }

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        # Create matrices efficiently using GPU when available
        exp_data = expected_output.map(&.to_f64)

        exp = if CUDA.fully_available?
                mat = CudaMatrix.new(1, exp_data.size)
                exp_data.each_with_index { |val, i| mat[0, i] = val }
                mat.sync_to_device!("evaluate_expected_matrix")
                mat
              else
                SimpleMatrix.from_a([exp_data])
              end

        act = if CUDA.fully_available?
                mat = CudaMatrix.new(1, actual_output.size)
                actual_output.each_with_index { |val, i| mat[0, i] = val }
                mat.sync_to_device!("evaluate_actual_matrix")
                mat
              else
                SimpleMatrix.from_a([actual_output])
              end

        diff = if act.is_a?(CudaMatrix) && exp.is_a?(CudaMatrix)
                 act - exp
               else
                 act_s = act.is_a?(CudaMatrix) ? act.to_simple : act
                 exp_s = exp.is_a?(CudaMatrix) ? exp.to_simple : exp
                 act_s - exp_s
               end
        out_w = @output_layers.last.weights
        w = out_w.is_a?(CudaMatrix) ? out_w : GPUMemory.keep_on_gpu(out_w.as(SimpleMatrix))
        trans = if diff.is_a?(CudaMatrix) && w.is_a?(CudaMatrix)
                  diff * w
                else
                  d = diff.is_a?(CudaMatrix) ? diff.to_simple : diff
                  ww = w.is_a?(CudaMatrix) ? w.to_simple : w
                  d * ww
                end
        @transformer_error = trans.is_a?(CudaMatrix) ? trans.to_simple : trans
      end

      # puts "@error_signal: #{@error_signal}"
      # puts "@total_error: #{@total_error}"


    rescue e : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{e}")
    end

    # Evaluate using matrices already on the desired device
    # GPU-compatible version that preserves matrix type
    def evaluate(input_data : SimpleMatrix | CudaMatrix,
                 expected_output : SimpleMatrix | CudaMatrix,
                 cost_function : CostFunction = SHAInet.quadratic_cost)
      actual_matrix = run(input_data, stealth: true)

      output_layer = @output_layers.last
      grad = GPUMemory.like(actual_matrix, actual_matrix.rows, actual_matrix.cols)

      loss_value = 0.0

      if actual_matrix.is_a?(CudaMatrix) && expected_output.is_a?(CudaMatrix) && CUDNN.available?
        begin
          CUDNN.softmax_cross_entropy_loss_and_gradient(
            actual_matrix.as(CudaMatrix),
            expected_output.as(CudaMatrix),
            pointerof(loss_value),
            grad.as(CudaMatrix)
          )
        rescue e
          loss_value = compute_cost_and_gradient_cpu(actual_matrix, expected_output, grad, cost_function)
        end
      else
        loss_value = compute_cost_and_gradient_cpu(actual_matrix, expected_output, grad, cost_function)
      end

      @error_signal = [loss_value]
      @total_error = loss_value

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        diff = grad
        out_w = @output_layers.last.weights
        w = out_w.is_a?(CudaMatrix) ? out_w.as(CudaMatrix) : GPUMemory.keep_on_gpu(out_w.as(SimpleMatrix)).as(CudaMatrix)
        trans = if diff.is_a?(CudaMatrix) && w.is_a?(CudaMatrix)
                  diff * w
                else
                  d = diff.is_a?(CudaMatrix) ? diff.to_simple : diff
                  ww = w.is_a?(CudaMatrix) ? w.to_simple : w
                  d * ww
                end
        @transformer_error = trans.is_a?(CudaMatrix) ? trans.to_simple : trans
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{e}")
    end

    # Accept integer input for embeddings
    # This is a convenience wrapper around the standard evaluate method
    def evaluate(input_data : Array(Int32),
                 expected_output : Array(GenNum),
                 cost_function : CostFunction = SHAInet.quadratic_cost)
      evaluate(input_data.map(&.to_f64), expected_output, cost_function)
    end

    def evaluate_sequence(input_data : Array(Array(GenNum)),
                          expected_output : Array(GenNum),
                          cost_function : CostFunction = SHAInet.quadratic_cost)
      seq = input_data.map do |x|
        @mixed_precision ? x.map { |v| v.to_f32 to_f64 } : x.map(&.to_f64)
      end
      outputs = run(seq, stealth: true)
      actual_output = outputs.last

      # Test for NaNs & exploading gradients
      validate_values(actual_output, "actual_output")

      # Get the error signal for the final layer, based on the cost function
      @error_signal = [] of Float64 # Collect all the errors for current run

      actual_output.size.times do |i|
        cost = cost_function.call(expected_output[i], actual_output[i])
        @error_signal << cost[:value]

        # puts "Actual output: #{actual_output}"
        # puts "Cost value: #{cost[:value]}"
        # puts "Cost derivative: #{cost[:derivative]}"
        # puts "---"
      end

      # Test for NaNs & exploading gradients
      validate_values(@error_signal, "error_signal")
      @total_error = @error_signal.reduce(0.0) { |acc, i| acc + i }

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        exp_row = GPUMemory.to_gpu(SimpleMatrix.from_a([expected_output.map(&.to_f64)]))
        act_row = GPUMemory.to_gpu(SimpleMatrix.from_a([actual_output]))
        diff = act_row - exp_row
        tmp = GPUMemory.zeros_like(diff, outputs.size, diff.cols)
        tmp = tmp.to_simple if tmp.is_a?(CudaMatrix)

        # Use efficient row copying instead of element-by-element access
        tmp.set_row!(outputs.size - 1, diff.is_a?(CudaMatrix) ? diff.to_simple : diff, 0)
        @transformer_error = tmp
      end

      # puts "@error_signal: #{@error_signal}"
      # puts "@total_error: #{@total_error}"


    rescue e : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{e}")
    end

    # Convenience wrapper for integer inputs
    def evaluate_sequence_label(input_data : Array(Array(Int32)), label : Int32)
      seq = input_data.map { |x| x.map(&.to_f64) }
      evaluate_sequence_label(seq, label)
    end

    # Evaluate a single example using a class label and softmax cross entropy
    def evaluate_label(input_data : Array(GenNum), label : Int32)
      processed = input_data.map(&.to_f64)
      matrix = GPUMemory.to_gpu(SimpleMatrix.from_a([processed]))
      logits = run(matrix, stealth: true)

      if logits.is_a?(CudaMatrix)
        if label < 0 || label >= logits.cols
          raise NeuralNetRunError.new("Label #{label} out of bounds for output size #{logits.cols}")
        end

        target = CudaMatrix.zeros(1, logits.cols)
        target[0, label] = 1.0
        target.sync_to_device!

        grad = CudaMatrix.new(1, logits.cols)
        loss_val = 0.0

        if CUDNN.available?
          CUDNN.softmax_cross_entropy_loss_and_gradient(logits.as(CudaMatrix), target, pointerof(loss_val), grad)
        else
          logits.as(CudaMatrix).softmax_rows!
          grad.copy_from!(logits.as(CudaMatrix))
          grad[0, label] = grad[0, label] - 1.0
          logits.as(CudaMatrix).sync_from_device!("eval_label")
          loss_val = -Math.log(logits.as(CudaMatrix).unsafe_get(0, label).clamp(1e-9, 1.0))
        end

        @error_signal = Array(Float64).new(logits.cols, 0.0)
        @error_signal[label] = loss_val
        @total_error = loss_val

        if @hidden_layers.any? &.is_a?(TransformerLayer)
          out_w = @output_layers.last.weights
          w = out_w.is_a?(CudaMatrix) ? out_w.as(CudaMatrix) : GPUMemory.keep_on_gpu(out_w.as(SimpleMatrix)).as(CudaMatrix)
          trans = grad * w
          @transformer_error = trans.is_a?(CudaMatrix) ? trans.to_simple : trans
        end
      else
        actual_output = logits.as(SimpleMatrix).to_a.first
        validate_values(actual_output, "actual_output")
        probs = SHAInet.softmax(actual_output)

        if label < 0 || label >= probs.size
          raise NeuralNetRunError.new("Label #{label} out of bounds for output size #{probs.size}")
        end

        @error_signal = [] of Float64
        probs.size.times do |i|
          @error_signal << (i == label ? -Math.log(probs[i].clamp(1e-9, 1.0)) : 0.0)
        end

        validate_values(@error_signal, "error_signal")
        @total_error = -Math.log(probs[label].clamp(1e-9, 1.0))
      end
    end

    # Convenience wrapper for integer inputs
    def evaluate_label(input_data : Array(Int32), label : Int32)
      evaluate_label(input_data.map(&.to_f64), label)
    end

    # Convenience wrapper for matrix inputs
    def evaluate_label(input_data : SimpleMatrix, label : Int32)
      vec = input_data.to_a.first.map(&.to_f64)
      evaluate_label(vec, label)
    end

    # Evaluate a sequence example with a class label and softmax cross entropy
    def evaluate_sequence_label(input_data : Array(Array(GenNum)), label : Int32)
      seq = input_data.map { |x| x.map(&.to_f64) }
      matrix = GPUMemory.to_gpu(SimpleMatrix.from_a(seq))
      logits = run(matrix, stealth: true)

      if logits.is_a?(CudaMatrix)
        if label < 0 || label >= logits.cols
          raise NeuralNetRunError.new("Label #{label} out of bounds for output size #{logits.cols}")
        end

        target = CudaMatrix.zeros(1, logits.cols)
        target[0, label] = 1.0
        target.sync_to_device!

        grad = CudaMatrix.new(1, logits.cols)
        loss_val = 0.0

        if CUDNN.available?
          CUDNN.softmax_cross_entropy_loss_and_gradient(logits.as(CudaMatrix), target, pointerof(loss_val), grad)
        else
          logits.as(CudaMatrix).softmax_rows!
          grad.copy_from!(logits.as(CudaMatrix))
          grad[0, label] = grad[0, label] - 1.0
          logits.as(CudaMatrix).sync_from_device!("eval_seq_label")
          loss_val = -Math.log(logits.as(CudaMatrix).unsafe_get(0, label).clamp(1e-9, 1.0))
        end

        @error_signal = Array(Float64).new(logits.cols, 0.0)
        @error_signal[label] = loss_val
        @total_error = loss_val

        if @hidden_layers.any? &.is_a?(TransformerLayer)
          out_w = @output_layers.last.weights
          w = out_w.is_a?(CudaMatrix) ? out_w.as(CudaMatrix) : GPUMemory.keep_on_gpu(out_w.as(SimpleMatrix)).as(CudaMatrix)
          trans = grad * w
          tmp = GPUMemory.zeros_like(trans, matrix.rows, trans.cols)
          tmp = tmp.to_simple if tmp.is_a?(CudaMatrix)
          trans_s = trans.is_a?(CudaMatrix) ? trans.to_simple : trans
          trans.cols.times do |j|
            tmp[matrix.rows - 1, j] = trans_s[0, j]
          end
          @transformer_error = tmp
        end
      else
        outputs = logits.as(SimpleMatrix).to_a
        actual_output = outputs.last
        validate_values(actual_output, "actual_output")
        probs = SHAInet.softmax(actual_output)

        if label < 0 || label >= probs.size
          raise NeuralNetRunError.new("Label #{label} out of bounds for output size #{probs.size}")
        end

        @error_signal = [] of Float64
        @total_error = -Math.log(probs[label].clamp(1e-9, 1.0))

        if @hidden_layers.any? &.is_a?(TransformerLayer)
          exp_row = SimpleMatrix.zeros(1, probs.size)
          exp_row[0, label] = 1.0
          act_row = SimpleMatrix.from_a([probs])
          diff = act_row - exp_row
          out_w = @output_layers.last.weights
          w = out_w.is_a?(CudaMatrix) ? out_w.to_simple : out_w.as(SimpleMatrix)
          trans = diff * w
          tmp = SimpleMatrix.zeros(matrix.rows, trans.cols)
          trans.cols.times do |j|
            tmp[matrix.rows - 1, j] = trans[0, j]
          end
          @transformer_error = tmp
        end
      end
    end

    # Convenience wrapper for integer inputs
    def evaluate_sequence_label(input_data : Array(Array(Int32)), label : Int32)
      seq = input_data.map { |x| x.map(&.to_f64) }
      evaluate_sequence_label(seq, label)
    end

    # Calculate MSE from the error signal of the output layer
    def update_mse
      n = @output_layers.last.size
      if @error_signal.size == 1
        error_avg = 0.0
      else
        error_avg = @total_error / n
      end
      sqrd_dists = 0.0
      @error_signal.each { |e| sqrd_dists += (e - error_avg)**2 }
      @mse = sqrd_dists / n
    end

    # Clean matrix-based training method
    def train(data : Array(Array) | SHAInet::TrainingData | SHAInet::StreamingData,
              training_type : Symbol | String = :sgdm,
              cost_function : Symbol | String | CostFunction = :mse,
              epochs : Int32 = 1,
              error_threshold : Float64 = 0.00000001,
              mini_batch_size : Int32 = 1,
              log_each : Int32 = 1,
              show_slice : Bool = false,
              autosave : NamedTuple(freq: Int32, path: String) | Nil = nil)
      verify_net_before_train

      stream = data.is_a?(SHAInet::StreamingData) ? data : nil
      if data.is_a?(SHAInet::TrainingData) && data.preload_gpu? && CUDA.fully_available?
        data.preload_gpu!
      end
      # Convert TrainingData to raw data array
      raw_data = if data.is_a?(SHAInet::TrainingData)
                   data.data
                 elsif data.is_a?(Array)
                   data.as(Array)
                 else
                   [] of Array(Array(Float64))
                 end

      # Validate and convert cost function
      if cost_function.is_a?(Symbol) || cost_function.is_a?(String)
        raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :c_ent_sm).") if COST_FUNCTIONS.any? { |x| x == cost_function.to_s } == false
        cost_proc = get_cost_proc(cost_function.to_s)
      else
        cost_proc = cost_function
      end

      if stream
        Log.info { "Training started with #{epochs} epochs (streaming)" }
      else
        Log.info { "Training started with #{epochs} epochs, #{raw_data.size} samples" }
      end
      start_time = Time.monotonic

      batch_size = stream ? mini_batch_size : mini_batch_size.clamp(1, raw_data.size)

      epochs.times do |epoch|
        # Reset sync counters at start of each epoch
        if CUDA.fully_available?
          SHAInet::CudaMatrix.reset_sync_stats
        end

        # Autosave if configured
        if autosave && epoch % autosave[:freq] == 0 && epoch > 0
          save_to_file("#{autosave[:path]}/autosave_epoch_#{epoch}.nn")
        end

        # Shuffle or rewind data for each epoch
        total_error = 0.0
        sample_count = 0

        if stream
          stream.rewind if epoch > 0
          while (batch = stream.next_batch(batch_size)).size > 0
            batch_error = process_batch(batch, cost_proc, training_type)
            total_error += batch_error
            sample_count += batch.size
          end
        else
          shuffled_data = raw_data.shuffle
          # Process data in mini-batches
          shuffled_data.each_slice(batch_size) do |batch|
            batch_error = process_batch(batch, cost_proc, training_type)
            total_error += batch_error
            sample_count += batch.size
          end
        end

        avg_error = total_error / sample_count
        @total_error = total_error
        @error_signal = [avg_error]
        update_mse

        if epoch % log_each == 0
          elapsed = Time.monotonic - start_time
          gpu_stats = SHAInet::CudaMatrix.gpu_memory_stats
          sync_stats = CUDA.fully_available? ? SHAInet::CudaMatrix.sync_stats : nil

          if sync_stats
            sync_to_mb = (sync_stats[:total_sync_bytes_to_device] / 1024.0 / 1024.0).round(2)
            sync_from_mb = (sync_stats[:total_sync_bytes_from_device] / 1024.0 / 1024.0).round(2)
            total_syncs = sync_stats[:sync_to_device_count] + sync_stats[:sync_from_device_count]
            Log.info { "Epoch: #{epoch}, Error: #{avg_error.round(6)}, MSE: #{@mse.round(6)}, Time: #{elapsed.total_seconds.round(2)}s" }
            Log.debug { "  GPU: #{gpu_stats[:active_matrices]} matrices, #{(gpu_stats[:total_allocated_bytes] / 1024.0 / 1024.0).round(2)} MB" }
            Log.debug { "  Syncs: #{total_syncs} total (#{sync_stats[:sync_to_device_count]} to GPU, #{sync_stats[:sync_from_device_count]} from GPU)" }
            Log.debug { "  Data: #{sync_to_mb} MB to GPU, #{sync_from_mb} MB from GPU (#{(sync_to_mb + sync_from_mb).round(2)} MB total)" }
            Log.debug { "  Matrix creations: #{sync_stats[:matrix_creation_count]} this epoch" }
            SHAInet::CudaMatrix.print_top_allocation_sites

            # Log top sync sources
            sources = SHAInet::CudaMatrix.sync_sources_stats
            if sources.size > 0
              Log.debug { "  Top sync sources:" }
              sources.to_a.sort_by { |k, v| v }.reverse[0, 5].each do |source, count|
                Log.debug { "    #{source}: #{count} times" }
              end
            end
          else
            Log.info { "Epoch: #{epoch}, Error: #{avg_error.round(6)}, MSE: #{@mse.round(6)}, Time: #{elapsed.total_seconds.round(2)}s, GPU: #{gpu_stats[:active_matrices]} matrices, #{gpu_stats[:total_allocated_bytes]} bytes" }
          end

          # Log matrix pool statistics
          if CUDA.fully_available?
            pool_stats = CudaMatrix.pool_stats
            Log.debug { "  Matrix pool: #{pool_stats[:total_pooled]} matrices pooled across #{pool_stats[:pools].size} sizes" }
            if pool_stats[:pools].size > 0
              top_pools = pool_stats[:pools].to_a.sort_by(&.[1]).reverse.first(3)
              Log.debug { "    Top pool sizes: #{top_pools.map { |k, v| "#{k}(#{v})" }.join(", ")}" }
            end
          end
        end

        if avg_error < error_threshold
          Log.info { "Training stopped early. Error threshold reached: #{avg_error} < #{error_threshold}" }
          break
        end

        @time_step += 1
      end

      elapsed = Time.monotonic - start_time
      Log.info { "Training completed in #{elapsed.total_seconds.round(2)} seconds" }
    end

    private def process_batch(batch, cost_proc, training_type)
      batch_error = 0.0

      # Zero gradients for all matrix layers
      @hidden_layers.each do |layer|
        layer.zero_gradients if layer.is_a?(MatrixLayer)
      end
      @output_layers.each do |layer|
        layer.zero_gradients if layer.is_a?(MatrixLayer)
      end

      # Determine matrix dimensions from first sample for workspace allocation
      first_input = batch.first[0]
      first_output = batch.first[1]
      # If the first output is a single label, expand to one-hot only when GPU
      # optimized label cross-entropy is unavailable.
      if first_output.is_a?(Array) && first_output.as(Array).size == 1 &&
         !first_output.as(Array)[0].is_a?(Array) && @output_layers.last.is_a?(MatrixLayer)
        if !(CUDA.fully_available? && CUDNN.available? &&
           @output_layers.last.as(MatrixLayer).size > 1)
          label = first_output.as(Array).first.as(GenNum).to_i
          oh = Array(Float64).new(@output_layers.last.as(MatrixLayer).size, 0.0)
          oh[label] = 1.0 if label >= 0 && label < oh.size
          first_output = oh
        end
      end

      get_dims = ->(obj : SimpleMatrix | CudaMatrix | Array(Array(Float64)) | Array(Float64)) do
        case obj
        when SimpleMatrix
          {obj.rows, obj.cols}
        when CudaMatrix
          {obj.rows, obj.cols}
        else
          arr = obj.as(Array)
          if arr.size > 0 && arr[0].is_a?(Array)
            {arr.size, arr[0].as(Array).size}
          else
            {1, arr.size}
          end
        end
      end

      in_rows, in_cols = get_dims.call(first_input)
      out_rows, out_cols = get_dims.call(first_output)

      input_workspace : CudaMatrix | Nil = nil
      expected_workspace : CudaMatrix | Nil = nil
      output_grad : SimpleMatrix | CudaMatrix | Nil = nil

      if CUDA.fully_available?
        if !first_input.is_a?(CudaMatrix)
          if @batch_in_ws.nil? || @batch_in_ws.not_nil!.rows != in_rows || @batch_in_ws.not_nil!.cols != in_cols
            @batch_in_ws = CudaMatrix.new(in_rows, in_cols)
          end
          input_workspace = @batch_in_ws
        end
        if !first_output.is_a?(CudaMatrix)
          if @batch_out_ws.nil? || @batch_out_ws.not_nil!.rows != out_rows || @batch_out_ws.not_nil!.cols != out_cols
            @batch_out_ws = CudaMatrix.new(out_rows, out_cols)
          end
          expected_workspace = @batch_out_ws
        end
        if @batch_grad_ws.nil? || @batch_grad_ws.not_nil!.rows != out_rows || @batch_grad_ws.not_nil!.cols != out_cols
          @batch_grad_ws = CudaMatrix.new(out_rows, out_cols)
        end
        output_grad = @batch_grad_ws
      end

      batch.each do |sample|
        input_data = sample[0]
        expected_output = sample[1]
        # If expected output is a single label, expand to one-hot only when GPU
        # accelerated label cross-entropy cannot be used.
        if expected_output.is_a?(Array) && expected_output.as(Array).size == 1 &&
           !expected_output.as(Array)[0].is_a?(Array) && @output_layers.last.is_a?(MatrixLayer)
          if !(CUDA.fully_available? && CUDNN.available? &&
             @output_layers.last.as(MatrixLayer).size > 1)
            label = expected_output.as(Array).first.as(GenNum).to_i
            oh = Array(Float64).new(@output_layers.last.as(MatrixLayer).size, 0.0)
            oh[label] = 1.0 if label >= 0 && label < oh.size
            expected_output = oh
          end
        end

        # Prepare expected output matrix using workspace when on GPU
        expected_matrix = case expected_output
                          when SimpleMatrix
                            if expected_workspace
                              GPUMemory.to_gpu!(expected_output, expected_workspace)
                            else
                              expected_output
                            end
                          when CudaMatrix
                            expected_output
                          else
                            arr = expected_output.as(Array)
                            if expected_workspace
                              case arr
                              when Array(Array(GenNum)), Array(Array(Float64))
                                # 2D: arr is Array(Array(GenNum)) or Array(Array(Float64))
                                GPUMemory.to_gpu!(arr.map { |row| row.map(&.to_f64) }, expected_workspace)
                              when Array(Float64)
                                # 1D: arr is Array(Float64)
                                GPUMemory.to_gpu!([arr], expected_workspace)
                              when Array(GenNum)
                                # 1D: arr is Array(GenNum)
                                GPUMemory.to_gpu!([arr.map(&.to_f64)], expected_workspace)
                              else
                                # Scalar fallback (should not happen, but for safety)
                                GPUMemory.to_gpu!([[arr.to_f64]], expected_workspace)
                              end
                              expected_workspace
                            else
                              to_matrix(expected_output)
                            end
                          end

        # Prepare input matrix using preallocated workspace when on GPU
        input_matrix = case input_data
                       when SimpleMatrix
                         if input_workspace
                           GPUMemory.to_gpu!(input_data, input_workspace)
                         else
                           input_data
                         end
                       when CudaMatrix
                         input_data
                       else
                         arr = input_data.as(Array)
                         if input_workspace
                           if arr.size > 0 && arr[0].is_a?(Array)
                             GPUMemory.to_gpu!(arr.as(Array(Array(Float64))), input_workspace)
                           else
                             GPUMemory.to_gpu!(arr.as(Array(Float64)), input_workspace)
                           end
                           input_workspace
                         else
                           to_matrix(input_data)
                         end
                       end

        actual_matrix = run(input_matrix, stealth: true)

        # Optimize: Use GPU-accelerated cost and gradient computation when possible
        sample_error = 0.0
        output_layer = @output_layers.last

        if output_layer.is_a?(MatrixLayer)
          # Determine dimensions for output_grad matrix
          use_label_gpu = actual_matrix.is_a?(CudaMatrix) && expected_matrix.is_a?(CudaMatrix) &&
                          CUDNN.available? && expected_matrix.as(CudaMatrix).cols == 1 &&
                          actual_matrix.as(CudaMatrix).cols > 1

          grad_rows, grad_cols = if use_label_gpu
                                   {actual_matrix.as(CudaMatrix).rows, actual_matrix.as(CudaMatrix).cols}
                                 elsif expected_output.is_a?(SimpleMatrix)
                                   exp_mat = expected_output.as(SimpleMatrix)
                                   {exp_mat.rows, exp_mat.cols}
                                 elsif expected_output.is_a?(CudaMatrix)
                                   exp_mat = expected_output.as(CudaMatrix)
                                   {exp_mat.rows, exp_mat.cols}
                                 elsif expected_output.is_a?(Array) && expected_output.as(Array).size > 0 && expected_output.as(Array)[0].is_a?(Array)
                                   {expected_output.as(Array).size, expected_output.as(Array)[0].as(Array).size}
                                 else
                                   arr = expected_output.as(Array)
                                   {1, arr.size}
                                 end

          # Reuse preallocated output_grad matrix when available
          if output_grad.nil? || output_grad.not_nil!.rows != grad_rows || output_grad.not_nil!.cols != grad_cols
            output_grad = GPUMemory.like(actual_matrix, grad_rows, grad_cols)
            @batch_grad_ws = output_grad if CUDA.fully_available? && output_grad.is_a?(CudaMatrix)
          end

          existing_grad = output_grad.not_nil!
          if existing_grad.is_a?(CudaMatrix)
            existing_grad.zero!
          else
            existing_grad.rows.times do |i|
              existing_grad.cols.times do |j|
                existing_grad[i, j] = 0.0
              end
            end
          end

          grad_matrix = output_grad.not_nil!

          # Try GPU-accelerated cross-entropy when possible
          if actual_matrix.is_a?(CudaMatrix) && expected_matrix.is_a?(CudaMatrix) && CUDNN.available?
            begin
              loss_value = 0.0
              if use_label_gpu
                CUDNN.softmax_cross_entropy_label_loss_and_gradient(
                  actual_matrix.as(CudaMatrix),
                  expected_matrix.as(CudaMatrix),
                  pointerof(loss_value),
                  grad_matrix.as(CudaMatrix)
                )
              else
                CUDNN.softmax_cross_entropy_loss_and_gradient(
                  actual_matrix.as(CudaMatrix),
                  expected_matrix.as(CudaMatrix),
                  pointerof(loss_value),
                  grad_matrix.as(CudaMatrix)
                )
              end
              sample_error = loss_value
            rescue e : Exception
              Log.debug { "GPU cross-entropy failed: #{e}, falling back to CPU computation" } unless use_label_gpu
              # Fall back to CPU computation below
              if use_label_gpu
                # Convert label indices to one-hot for CPU fallback
                one_hot = SimpleMatrix.zeros(expected_matrix.rows, actual_matrix.cols)
                expected_matrix.rows.times do |i|
                  label = expected_matrix.as(CudaMatrix).unsafe_get(i, 0).to_i
                  one_hot[i, label] = 1.0 if label >= 0 && label < actual_matrix.cols
                end
                sample_error = compute_cost_and_gradient_cpu(actual_matrix, one_hot, grad_matrix, cost_proc)
              else
                sample_error = compute_cost_and_gradient_cpu(actual_matrix, expected_matrix, grad_matrix, cost_proc)
              end
            end
          else
            # CPU fallback for non-CudaMatrix types or when GPU computation is unavailable
            if use_label_gpu
              one_hot = SimpleMatrix.zeros(expected_matrix.rows, actual_matrix.cols)
              expected_matrix.rows.times do |i|
                label = expected_matrix.as(CudaMatrix).unsafe_get(i, 0).to_i
                one_hot[i, label] = 1.0 if label >= 0 && label < actual_matrix.cols
              end
              sample_error = compute_cost_and_gradient_cpu(actual_matrix, one_hot, grad_matrix, cost_proc)
            else
              sample_error = compute_cost_and_gradient_cpu(actual_matrix, expected_matrix, grad_matrix, cost_proc)
            end
          end

          batch_error += sample_error

          # grad_matrix is already GPU-compatible and reused across samples
          grad = output_layer.backward(grad_matrix)

          # Handle transformer layers backward pass with proper gradient reshaping
          if @transformer_layers.any?
            # For transformers, we need to map gradients from output space back to transformer space
            # The gradient from output layer is (1 x vocab_size), but transformer expects (seq_len x d_model)

            # Get the transformer's d_model dimension from layer size
            d_model = @transformer_layers.first.size
            seq_len = 16 # hardcoded for now, should be dynamic

            if grad.rows == 1 && grad.cols != d_model
              # We have output gradients (1 x vocab_size), need to transform to (seq_len x d_model)
              # This requires going through the output layer weights transpose

              # Use appropriate matrix type based on whether grad is CudaMatrix or SimpleMatrix
              if grad.is_a?(CudaMatrix)
                output_weights = @output_layers.last.weights.as(CudaMatrix)
              else
                output_weights = @output_layers.last.weights.as(SimpleMatrix)
              end

              # Transform: (1 x vocab_size) * (vocab_size x d_model) -> (1 x d_model)
              if grad.cols == output_weights.rows
                if grad.is_a?(CudaMatrix)
                  transformed_grad = grad * output_weights.as(CudaMatrix).transpose
                else
                  transformed_grad = grad * output_weights.as(SimpleMatrix).transpose
                end
              else
                # Fallback: create zero gradients with correct dimensions
                mat_klass = grad.is_a?(CudaMatrix) ? CudaMatrix : SimpleMatrix
                transformed_grad = mat_klass.zeros(1, d_model)
              end

              # Check if we can reuse cached gradient matrix
              if @cached_expanded_grad.nil? || @cached_seq_len != seq_len || @cached_d_model != d_model ||
                 @cached_expanded_grad.not_nil!.class != transformed_grad.class
                # Create new cached matrix
                @cached_expanded_grad = if transformed_grad.is_a?(CudaMatrix)
                                          CudaMatrix.zeros(seq_len, d_model)
                                        else
                                          SimpleMatrix.zeros(seq_len, d_model)
                                        end
                @cached_seq_len = seq_len
                @cached_d_model = d_model
              end

              # Reuse cached matrix - zero only the last row that we'll update
              expanded_grad = @cached_expanded_grad.not_nil!
              # Zero out only the last row efficiently
              d_model.times do |j|
                expanded_grad[seq_len - 1, j] = transformed_grad[0, j]
              end

              grad = expanded_grad
            end

            @transformer_layers.reverse_each do |layer|
              grad = layer.backward(grad)
            end
          end

          @hidden_layers.reverse_each do |layer|
            if layer.is_a?(MatrixLayer)
              grad = layer.backward(grad)
            elsif layer.is_a?(EmbeddingLayer)
              layer.accumulate_gradient
            end
          end
        end

        # Explicit GPU memory cleanup after processing each sample
        # Force cleanup of temporary matrices created during forward/backward pass
        # Persistent GPU buffers handle workspace reuse

        # Return workspace matrices used for this sample
        if input_matrix.is_a?(CudaMatrix) && input_workspace && input_matrix.object_id == input_workspace.object_id
          CudaMatrix.return_workspace(input_matrix)
        end
        if expected_matrix.is_a?(CudaMatrix) && expected_workspace && expected_matrix.object_id == expected_workspace.object_id
          CudaMatrix.return_workspace(expected_matrix)
        end
      end

      learning_rate = current_learning_rate

      @hidden_layers.each do |layer|
        if layer.is_a?(MatrixLayer)
          layer.update_weights(learning_rate)
        elsif layer.is_a?(EmbeddingLayer)
          layer.apply_gradients(learning_rate)
        end
      end

      @output_layers.each do |layer|
        layer.update_weights(learning_rate) if layer.is_a?(MatrixLayer)
      end

      update_transformer_layers if @transformer_layers.any?

      batch_error
    end

    # Convert raw data (arrays) to matrix format
    # This method is only called when input is NOT already a matrix
    private def to_matrix(obj) : SimpleMatrix | CudaMatrix
      arr = obj.as(Array)

      # When CUDA is available, allocate CudaMatrix directly and
      # populate it using unsafe_set to avoid an intermediate
      # SimpleMatrix allocation.
      if CUDA.fully_available?
        if arr.size > 0 && arr[0].is_a?(Array)
          rows = arr.size
          cols = arr[0].as(Array).size
          mat = CudaMatrix.new(rows, cols)
          rows.times do |i|
            cols.times do |j|
              mat.unsafe_set(i, j, arr[i].as(Array)[j].as(GenNum).to_f64)
            end
          end
          mat.sync_to_device!
          mat
        else
          cols = arr.size
          mat = CudaMatrix.new(1, cols)
          cols.times do |i|
            mat.unsafe_set(0, i, arr[i].as(GenNum).to_f64)
          end
          mat.sync_to_device!
          mat
        end
      else
        # CPU-only fallback uses SimpleMatrix as before
        mat = if arr.size > 0 && arr[0].is_a?(Array)
                rows = arr.size
                cols = arr[0].as(Array).size
                SimpleMatrix.new(rows, cols).tap do |m|
                  rows.times do |i|
                    cols.times do |j|
                      m[i, j] = arr[i].as(Array)[j].as(GenNum).to_f64
                    end
                  end
                end
              else
                SimpleMatrix.new(1, arr.size).tap do |m|
                  arr.size.times do |i|
                    m[0, i] = arr[i].as(GenNum).to_f64
                  end
                end
              end
        mat
      end
    end

    def current_learning_rate
      if @warmup_steps > 0 && @time_step < @warmup_steps
        @learning_rate * (@time_step.to_f64 / @warmup_steps)
      else
        @learning_rate
      end
    end

    def update_transformer_layers
      lr = current_learning_rate
      @transformer_layers.each do |layer|
        layer.apply_gradients(lr)
      end
    end

    def validate_values(array : Array(Float64), location : String)
      # Detect NaNs in output
      array.each { |ar| raise NeuralNetRunError.new(
        "Found a NaN value, run stopped.\n#{location}: #{array}") if ar.nan? }
    end

    def get_cost_proc(function_name : String) : CostFunction
      case function_name
      when "mse"
        SHAInet.quadratic_cost
      when "c_ent"
        # raise MathError.new("Cross entropy cost is not implemented fully yet, please use quadratic cost for now.")
        SHAInet.cross_entropy_cost
      when "c_ent_sm"
        SHAInet.cross_entropy_cost
      else
        raise NeuralNetInitalizationError.new("Must choose correct cost function or provide a correct Proc")
      end
    end

    # Evaluate the network performance on a test set
    def test(test_set)
      correct = 0
      incorrect = 0
      test_set.normalized_inputs.each_with_index do |input, index|
        output_array = run(input: input, stealth: true)
        if test_set.label_for_array(output_array) == test_set.label_for_array(test_set.normalized_outputs[index])
          correct += 1
        else
          incorrect += 1
        end
      end
      Log.info { "Predicted #{correct} out of #{correct + incorrect} (#{(correct.to_f/(correct + incorrect).to_f)*100}% accuracy)" }
      correct.to_f/(correct + incorrect).to_f
    end

    # GPU path - all CudaMatrix operations
    private def safe_output_transform(matrix : CudaMatrix, weights : CudaMatrix) : CudaMatrix
      begin
        # For transformer architectures, use only the last token's representation
        if @hidden_layers.any? &.is_a?(TransformerLayer)
          # Extract last token (row) from transformer output for language modeling using GPU kernel
          last_token = if CUDA.fully_available? && (mptr = matrix.device_ptr) && (wptr = weights.device_ptr) && !mptr.null? && !wptr.null?
                         begin
                           # Use more efficient GPU slice operation for last row
                           result = CudaMatrix.new(1, matrix.cols)
                           # Copy last row using GPU memory copy
                           last_row_offset = (matrix.rows - 1) * matrix.cols
                           CUDA.copy_device_to_device(
                             result.device_ptr.not_nil!,
                             mptr + last_row_offset,
                             (matrix.cols * 8).to_u64
                           )
                           result.mark_device_dirty!
                           result
                         rescue e
                           # Fallback to elementwise copy if GPU operation fails
                           last_token_fallback = CudaMatrix.new(1, matrix.cols)
                           matrix.cols.times do |j|
                             last_token_fallback[0, j] = matrix[matrix.rows - 1, j]
                           end
                           last_token_fallback.sync_to_device!
                           last_token_fallback
                         end
                       else
                         # CPU fallback
                         last_token_cpu = CudaMatrix.new(1, matrix.cols)
                         matrix.cols.times do |j|
                           last_token_cpu[0, j] = matrix[matrix.rows - 1, j]
                         end
                         last_token_cpu.sync_to_device!
                         last_token_cpu
                       end

          # Now multiply: last_token (1 x d_model) * weights (d_model x vocab_size)
          if last_token.cols != weights.rows
            raise ArgumentError.new("Transformer output dimension mismatch: d_model (#{last_token.cols}) doesn't match weights input size (#{weights.rows})")
          end

          return last_token * weights
        end

        # For matrix * weights, we need matrix.cols == weights.rows
        if matrix.cols != weights.rows
          raise ArgumentError.new("Matrix dimension mismatch: input features (#{matrix.cols}) doesn't match weights input size (#{weights.rows})")
        end

        matrix * weights
      rescue ex : Exception
        # Check for dimension issues and try to reshape if possible
        if ex.message.to_s.includes?("size mismatch") || ex.message.to_s.includes?("dimension mismatch")
          # Try reshaping for a single token/sequence case
          if matrix.rows == 1 && matrix.cols > 0 && weights.rows > 0 && weights.cols > 0
            Log.info { "Reshaping matrix for single-token transformer operation" }
            reshaped = CudaMatrix.new(1, weights.cols)
            weights.cols.times do |j|
              sum = 0.0
              matrix.cols.times do |k|
                sum += matrix[0, k] * weights[j, k]
              end
              reshaped[0, j] = sum
            end
            reshaped.sync_to_device!
            return reshaped
          end
        end

        raise ex
      end
    end

    # CPU path - all SimpleMatrix operations
    private def safe_output_transform(matrix : SimpleMatrix, weights : SimpleMatrix) : SimpleMatrix
      begin
        # For transformer architectures, use only the last token's representation
        if @hidden_layers.any? &.is_a?(TransformerLayer)
          # Extract last token (row) from transformer output for language modeling
          last_token = SimpleMatrix.new(1, matrix.cols)
          matrix.cols.times do |j|
            last_token[0, j] = matrix[matrix.rows - 1, j]
          end

          # Now multiply: last_token (1 x d_model) * weights (d_model x vocab_size)
          if last_token.cols != weights.rows
            raise ArgumentError.new("Transformer output dimension mismatch: d_model (#{last_token.cols}) doesn't match weights input size (#{weights.rows})")
          end

          return last_token * weights
        end

        # For matrix * weights, we need matrix.cols == weights.rows
        if matrix.cols != weights.rows
          raise ArgumentError.new("Matrix dimension mismatch: input features (#{matrix.cols}) doesn't match weights input size (#{weights.rows})")
        end

        matrix * weights
      rescue ex : Exception
        # Check for dimension issues and try to reshape if possible
        if ex.message.to_s.includes?("size mismatch") || ex.message.to_s.includes?("dimension mismatch")
          # Try reshaping for a single token/sequence case
          if matrix.rows == 1 && matrix.cols > 0 && weights.rows > 0 && weights.cols > 0
            Log.info { "Reshaping matrix for single-token transformer operation" }
            reshaped = SimpleMatrix.new(1, weights.cols)
            weights.cols.times do |j|
              sum = 0.0
              matrix.cols.times do |k|
                sum += matrix[0, k] * weights[j, k]
              end
              reshaped[0, j] = sum
            end
            return reshaped
          end
        end

        raise ex
      end
    end # Optimized helper to extract tokens from GPU matrix without elementwise access
    # Uses GPU-to-CPU batch transfer instead of per-element sync
    private def extract_tokens_gpu(matrix : CudaMatrix) : Array(Int32)
      # Sync entire matrix from GPU in one operation instead of elementwise access
      matrix.sync_from_device!("extract_tokens") if matrix.device_dirty?
      # Extract tokens as a batch operation from column 0
      Array.new(matrix.rows) { |r| matrix.unsafe_get(r, 0).to_i }
    end

    # Optimized matrix creation from arrays using batch operations
    private def create_matrix_from_arrays(data : Array(Array(Float64)), use_gpu : Bool = true) : SimpleMatrix | CudaMatrix
      return SimpleMatrix.from_a(data) unless use_gpu && CUDA.fully_available?

      # Create GPU matrix directly from array data in batch
      rows = data.size
      cols = data[0].size
      result = CudaMatrix.new(rows, cols)

      # Copy data in batch instead of elementwise
      flat_data = data.flatten
      result.raw_data.copy_from(flat_data.to_unsafe, flat_data.size)
      result.sync_to_device!
      result
    end

    # Optimized matrix population from single array using batch operations
    private def populate_matrix_batch(matrix : CudaMatrix | SimpleMatrix, data : Array(Float64), row : Int32)
      if matrix.is_a?(CudaMatrix)
        # Use raw data access for batch copy instead of elementwise assignment
        start_idx = row * matrix.cols
        matrix.raw_data.to_slice[start_idx, data.size].copy_from(data.to_unsafe, data.size)
        matrix.sync_to_device!
      else
        # For SimpleMatrix, still do elementwise but at least batch the operation
        data.each_with_index { |val, col| matrix[row, col] = val }
      end
    end

    # Try to apply activation function using GPU kernels
    private def try_gpu_activation(matrix : CudaMatrix, activation_function : ActivationFunction) : Bool
      return false unless CUDA.fully_available?

      case activation_function
      when SHAInet.sigmoid
        # Use in-place GPU sigmoid operation
        begin
          matrix.sigmoid!
          return true
        rescue e
          Log.debug { "GPU sigmoid failed: #{e}, falling back to CPU" }
        end
      when SHAInet.relu
        # Use in-place ReLU operation
        begin
          matrix.relu!
          return true
        rescue e
          Log.debug { "GPU ReLU failed: #{e}, falling back to CPU" }
        end
      end

      false
    end

    # Helper method for matrix slicing (missing method)
    private def slice_rows_helper(matrix : CudaMatrix, start_row : Int32, num_rows : Int32) : CudaMatrix
      result = CudaMatrix.new(num_rows, matrix.cols)
      num_rows.times do |i|
        matrix.cols.times do |j|
          result[i, j] = matrix[start_row + i, j]
        end
      end
      result.sync_to_device! if CUDA.fully_available?
      result
    end

    # CPU fallback for cost and gradient computation when GPU acceleration fails
    private def compute_cost_and_gradient_cpu(actual_matrix, expected_output, grad_matrix, cost_proc)
      sample_error = 0.0

      if actual_matrix.is_a?(CudaMatrix)
        actual_matrix.as(CudaMatrix).sync_from_device!("cost_grad_cpu")
      end

      if expected_output.is_a?(CudaMatrix)
        expected_output.as(CudaMatrix).sync_from_device!("cost_grad_cpu")
      end

      if expected_output.is_a?(SimpleMatrix)
        exp_mat = expected_output.as(SimpleMatrix)
        exp_mat.rows.times do |i|
          exp_mat.cols.times do |j|
            expected = exp_mat[i, j]
            actual = actual_matrix.is_a?(CudaMatrix) ? actual_matrix.as(CudaMatrix).unsafe_get(i, j) : actual_matrix.as(SimpleMatrix)[i, j]
            cost_result = cost_proc.call(expected, actual)
            sample_error += cost_result[:value]
            if grad_matrix.is_a?(CudaMatrix)
              grad_matrix.as(CudaMatrix).unsafe_set(i, j, cost_result[:derivative])
            else
              grad_matrix.as(SimpleMatrix)[i, j] = cost_result[:derivative]
            end
          end
        end
      elsif expected_output.is_a?(CudaMatrix)
        exp_mat = expected_output.as(CudaMatrix)
        exp_mat.rows.times do |i|
          exp_mat.cols.times do |j|
            expected = exp_mat.unsafe_get(i, j)
            actual = actual_matrix.is_a?(CudaMatrix) ? actual_matrix.as(CudaMatrix).unsafe_get(i, j) : actual_matrix.as(SimpleMatrix)[i, j]
            cost_result = cost_proc.call(expected, actual)
            sample_error += cost_result[:value]
            if grad_matrix.is_a?(CudaMatrix)
              grad_matrix.as(CudaMatrix).unsafe_set(i, j, cost_result[:derivative])
            else
              grad_matrix.as(SimpleMatrix)[i, j] = cost_result[:derivative]
            end
          end
        end
      elsif expected_output.is_a?(Array) && expected_output.as(Array).size > 0 && expected_output.as(Array)[0].is_a?(Array)
        rows = expected_output.as(Array).size
        cols = expected_output.as(Array)[0].as(Array).size
        rows.times do |i|
          cols.times do |j|
            expected = expected_output.as(Array)[i].as(Array)[j].as(GenNum).to_f64
            actual = actual_matrix.is_a?(CudaMatrix) ? actual_matrix.as(CudaMatrix).unsafe_get(i, j) : actual_matrix.as(SimpleMatrix)[i, j]
            cost_result = cost_proc.call(expected, actual)
            sample_error += cost_result[:value]
            if grad_matrix.is_a?(CudaMatrix)
              grad_matrix.as(CudaMatrix).unsafe_set(i, j, cost_result[:derivative])
            else
              grad_matrix.as(SimpleMatrix)[i, j] = cost_result[:derivative]
            end
          end
        end
      else
        arr = expected_output.as(Array)
        arr.size.times do |i|
          expected = arr[i].as(GenNum).to_f64
          actual = actual_matrix.is_a?(CudaMatrix) ? actual_matrix.as(CudaMatrix).unsafe_get(0, i) : actual_matrix.as(SimpleMatrix)[0, i]
          cost_result = cost_proc.call(expected, actual)
          sample_error += cost_result[:value]
          if grad_matrix.is_a?(CudaMatrix)
            grad_matrix.as(CudaMatrix).unsafe_set(0, i, cost_result[:derivative])
          else
            grad_matrix.as(SimpleMatrix)[0, i] = cost_result[:derivative]
          end
        end
      end

      if grad_matrix.is_a?(CudaMatrix)
        grad_matrix.as(CudaMatrix).sync_to_device!("cost_grad_cpu")
        grad_matrix.as(CudaMatrix).mark_device_dirty!
      end

      sample_error
    end
  end
end
