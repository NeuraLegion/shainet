require "log"
require "json"
require "../cuda"
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

    # Run an input throught the network to get an output (weights & biases do not change)
    def run(input : Array(GenNum), stealth : Bool = false) : Array(Float64)
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      raise NeuralNetRunError.new(
        "Error input data size: #{input.size} doesn't fit input layer size: #{expected_size}.") unless input.size == expected_size

      processed = @mixed_precision ? input.map { |v| v.to_f32.to_f64 } : input.map(&.to_f64)

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        matrix = GPUMemory.to_gpu(SimpleMatrix.from_a([processed]))
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            token = input.first.to_i
            if CUDA.available?
              matrix = l.as(EmbeddingLayer).embed([token])
            else
              vec = l.as(EmbeddingLayer).embed(token)
              matrix = GPUMemory.to_gpu(SimpleMatrix.from_a([vec]))
            end
          when TransformerLayer
            matrix = l.as(TransformerLayer).forward(matrix)
          end
        end
        out_layer = @output_layers.last
        w = GPUMemory.keep_on_gpu(out_layer.weights)
        b = GPUMemory.keep_on_gpu(out_layer.biases)
        matrix = safe_output_transform(matrix, w)

        # Delegate output computation to the matrix-based layer implementation
        matrix = out_layer.forward(matrix)

        output = matrix.to_a.first
        unless stealth
          Log.info { "Input => #{input}, network output => #{output}" }
        end
        output
      elsif @hidden_layers.none? { |l| l.is_a?(RecurrentLayer) || l.is_a?(LSTMLayer) }
        matrix = GPUMemory.to_gpu(SimpleMatrix.from_a([processed]))
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            token = input.first.to_i
            if CUDA.available?
              matrix = l.as(EmbeddingLayer).embed([token])
            else
              vec = l.as(EmbeddingLayer).embed(token)
              matrix = GPUMemory.to_gpu(SimpleMatrix.from_a([vec]))
            end
          when MatrixLayer
            matrix = l.as(MatrixLayer).forward(matrix)
          else
            matrix = l.forward(matrix)
          end
        end
        out_layer = @output_layers.last
        if out_layer.is_a?(MatrixLayer)
          matrix = out_layer.as(MatrixLayer).forward(matrix)
        else
          matrix = out_layer.forward(matrix)
        end
        output = matrix.to_a.first
        unless stealth
          Log.info { "Input => #{input}, network output => #{output}" }
        end
        output
      else
        # Insert the input data into the input layers
        index = 0

        # Propogate the information forward through the hidden layers

        @hidden_layers.each do |l|
          if l.is_a?(RecurrentLayer)
            l.as(RecurrentLayer).activate_step
          elsif l.is_a?(LSTMLayer)
            l.as(LSTMLayer).activate_step
          end
        end

        output = [] of Float64
        unless stealth
          Log.info { "Input => #{input}, network output => #{output}" }
        end
        output
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # Run using a pre-constructed matrix. Useful when batches are already on the GPU.
    def run(input : SimpleMatrix, stealth : Bool = false) : SimpleMatrix
      verify_net_before_train

      matrix = GPUMemory.keep_on_gpu(input)

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = (0...matrix.rows).map { |r| matrix[r, 0].to_i }
            matrix = l.as(EmbeddingLayer).embed(tokens)
          when TransformerLayer
            matrix = l.as(TransformerLayer).forward(matrix)
          end
        end
        out_layer = @output_layers.last
        w = GPUMemory.keep_on_gpu(out_layer.weights)
        b = GPUMemory.keep_on_gpu(out_layer.biases)
        matrix = safe_output_transform(matrix, w)
        matrix.rows.times do |i|
          matrix.cols.times do |j|
            matrix[i, j] += b[j, 0]
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
        matrix
      elsif @hidden_layers.none? { |l| l.is_a?(RecurrentLayer) || l.is_a?(LSTMLayer) }
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
        matrix
      else
        raise NeuralNetRunError.new("Matrix input not supported for recurrent networks")
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # Run a batch of sequences by calling `run` for each sequence
    # This is a convenience wrapper that can be consolidated with more direct matrix operations
    def run(input : Array(Array(Array(GenNum))), stealth : Bool = false) : Array(Array(Array(Float64)))
      input.map { |seq| run(seq, stealth: stealth) }
    end

    # Accept a sequence of integer tokens for embedding layers
    # This is a convenience wrapper around the standard run method
    def run(input : Array(Array(Int32)), stealth : Bool = false) : Array(Array(Float64))
      seq = input.map { |x| x.map(&.to_f64) }
      run(seq, stealth: stealth)
    end

    # Accept integer input for embedding layers
    # This is a convenience wrapper around the standard run method
    def run(input : Array(Int32), stealth : Bool = false) : Array(Float64)
      float_in = input.map(&.to_f64)
      run(float_in, stealth: stealth)
    end

    def run(input : Array(Array(GenNum)), stealth : Bool = false) : Array(Array(Float64))
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      input.each do |step|
        raise NeuralNetRunError.new("Error input data size: #{step.size} doesn't fit input layer size: #{expected_size}.") unless step.size == expected_size
      end
      processed = input.map do |x|
        @mixed_precision ? x.map { |v| v.to_f32.to_f64 } : x.map(&.to_f64)
      end
      if @hidden_layers.any? &.is_a?(TransformerLayer)
        matrix = GPUMemory.to_gpu(SimpleMatrix.from_a(processed))
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = (0...matrix.rows).map { |r| matrix[r, 0].to_i }
            if CUDA.available?
              matrix = l.as(EmbeddingLayer).embed(tokens)
            else
              embeddings = tokens.map { |id| l.as(EmbeddingLayer).embed(id) }
              matrix = GPUMemory.to_gpu(SimpleMatrix.from_a(embeddings))
            end
          when TransformerLayer
            matrix = l.as(TransformerLayer).forward(matrix)
          end
        end
        out_layer = @output_layers.last
        w = GPUMemory.keep_on_gpu(out_layer.weights)
        b = GPUMemory.keep_on_gpu(out_layer.biases)
        matrix = safe_output_transform(matrix, w)

        # Use GPU-accelerated bias addition when available
        if matrix.is_a?(CudaMatrix) && b.is_a?(CudaMatrix)
          matrix.add_bias!(b)
        else
          matrix.rows.times do |i|
            matrix.cols.times do |j|
              matrix[i, j] += b[j, 0]
            end
          end
        end

        # Apply activation function - for identity, no operation needed
        unless out_layer.activation_function == SHAInet.identity
          matrix.rows.times do |i|
            matrix.cols.times do |j|
              val = matrix[i, j]
              act, sig = out_layer.activation_function.call(val)
              matrix[i, j] = act
            end
          end
        end
        matrix.to_a
      else
        matrix = GPUMemory.to_gpu(SimpleMatrix.from_a(processed))
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = (0...matrix.rows).map { |r| matrix[r, 0].to_i }
            if CUDA.available?
              matrix = l.as(EmbeddingLayer).embed(tokens)
            else
              embeddings = tokens.map { |id| l.as(EmbeddingLayer).embed(id) }
              matrix = GPUMemory.to_gpu(SimpleMatrix.from_a(embeddings))
            end
          else
            matrix = l.forward(matrix)
          end
        end
        out_layer = @output_layers.last
        matrix = out_layer.forward(matrix)
        matrix.to_a
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
        exp = GPUMemory.to_gpu(SimpleMatrix.from_a([expected_output.map(&.to_f64)]))
        act = GPUMemory.to_gpu(SimpleMatrix.from_a([actual_output]))
        diff = act - exp
        w = GPUMemory.keep_on_gpu(@output_layers.last.weights)
        @transformer_error = diff * w
      end

      # puts "@error_signal: #{@error_signal}"
      # puts "@total_error: #{@total_error}"


    rescue e : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{e}")
    end

    # Evaluate using matrices already on the desired device
    def evaluate(input_data : SimpleMatrix,
                 expected_output : SimpleMatrix,
                 cost_function : CostFunction = SHAInet.quadratic_cost)
      actual_matrix = run(input_data, stealth: true)
      exp = expected_output.to_a.first
      act = actual_matrix.to_a.first
      validate_values(act, "actual_output")

      @error_signal = [] of Float64
      act.size.times do |i|
        # For matrix-based implementation, store gradients in matrices
        cost = cost_function.call(exp[i], act[i])

        # Matrix-based approach - store gradient in activation matrix
        @output_layers.last.activations[0, i] = cost[:derivative]*@output_layers.last.sigma_primes[0, i]

        @error_signal << cost[:value]
      end

      validate_values(@error_signal, "error_signal")
      @total_error = @error_signal.reduce(0.0) { |acc, i| acc + i }

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        exp_m = GPUMemory.to_gpu(SimpleMatrix.from_a([exp]))
        act_m = GPUMemory.to_gpu(SimpleMatrix.from_a([act]))
        diff = act_m - exp_m
        w = GPUMemory.keep_on_gpu(@output_layers.last.weights)
        @transformer_error = diff * w
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
        @mixed_precision ? x.map { |v| v.to_f32.to_f64 } : x.map(&.to_f64)
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
        @transformer_error = GPUMemory.zeros_like(diff, outputs.size, diff.cols)
        diff.cols.times do |j|
          @transformer_error[outputs.size - 1, j] = diff[0, j]
        end
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
      actual_output = run(input_data.map(&.to_f64), stealth: true)
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

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        exp = GPUMemory.to_gpu(SimpleMatrix.zeros(1, probs.size))
        exp[0, label] = 1.0
        act = GPUMemory.to_gpu(SimpleMatrix.from_a([probs]))
        diff = act - exp
        w = GPUMemory.keep_on_gpu(@output_layers.last.weights)
        @transformer_error = diff * w
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
      outputs = run(seq, stealth: true)
      actual_output = outputs.last
      validate_values(actual_output, "actual_output")
      probs = SHAInet.softmax(actual_output)

      if label < 0 || label >= probs.size
        raise NeuralNetRunError.new("Label #{label} out of bounds for output size #{probs.size}")
      end

      @error_signal = [] of Float64

      validate_values(@error_signal, "error_signal")
      @total_error = -Math.log(probs[label].clamp(1e-9, 1.0))

      if @hidden_layers.any? &.is_a?(TransformerLayer)
        exp_row = GPUMemory.to_gpu(SimpleMatrix.zeros(1, probs.size))
        exp_row[0, label] = 1.0
        act_row = GPUMemory.to_gpu(SimpleMatrix.from_a([probs]))
        diff = act_row - exp_row
        w = GPUMemory.keep_on_gpu(@output_layers.last.weights)
        trans = diff * w
        @transformer_error = GPUMemory.zeros_like(trans, outputs.size, trans.cols)
        trans.cols.times do |j|
          @transformer_error[outputs.size - 1, j] = trans[0, j]
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
          Log.info { "Epoch: #{epoch}, Error: #{avg_error.round(6)}, MSE: #{@mse.round(6)}, Time: #{elapsed.total_seconds.round(2)}s" }
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

      batch.each do |sample|
        input_data = sample[0]
        expected_output = sample[1]

        input_matrix = to_matrix(input_data)

        actual_matrix = run(input_matrix, stealth: true)

        sample_error = 0.0

        if expected_output.is_a?(SimpleMatrix)
          rows = expected_output.as(SimpleMatrix).rows
          cols = expected_output.as(SimpleMatrix).cols
          rows.times do |i|
            cols.times do |j|
              expected = expected_output.as(SimpleMatrix)[i, j]
              actual = actual_matrix[i, j]
              cost_result = cost_proc.call(expected, actual)
              sample_error += cost_result[:value]
            end
          end
        elsif expected_output.is_a?(Array) && expected_output.as(Array).size > 0 && expected_output.as(Array)[0].is_a?(Array)
          rows = expected_output.as(Array).size
          cols = expected_output.as(Array)[0].as(Array).size
          rows.times do |i|
            cols.times do |j|
              expected = expected_output.as(Array)[i].as(Array)[j].as(GenNum).to_f64
              actual = actual_matrix[i, j]
              cost_result = cost_proc.call(expected, actual)
              sample_error += cost_result[:value]
            end
          end
        else
          arr = expected_output.as(Array)
          arr.size.times do |i|
            expected = arr[i].as(GenNum).to_f64
            actual = actual_matrix[0, i]
            cost_result = cost_proc.call(expected, actual)
            sample_error += cost_result[:value]
          end
        end

        batch_error += sample_error

        output_layer = @output_layers.last
        if output_layer.is_a?(MatrixLayer)
          if expected_output.is_a?(SimpleMatrix)
            exp_mat = expected_output.as(SimpleMatrix)
            output_grad = SimpleMatrix.new(exp_mat.rows, exp_mat.cols)
            exp_mat.rows.times do |i|
              exp_mat.cols.times do |j|
                expected = exp_mat[i, j]
                actual = actual_matrix[i, j]
                cost_result = cost_proc.call(expected, actual)
                output_grad[i, j] = cost_result[:derivative]
              end
            end
          elsif expected_output.is_a?(Array) && expected_output.as(Array).size > 0 && expected_output.as(Array)[0].is_a?(Array)
            rows = expected_output.as(Array).size
            cols = expected_output.as(Array)[0].as(Array).size
            output_grad = SimpleMatrix.new(rows, cols)
            rows.times do |i|
              cols.times do |j|
                expected = expected_output.as(Array)[i].as(Array)[j].as(GenNum).to_f64
                actual = actual_matrix[i, j]
                cost_result = cost_proc.call(expected, actual)
                output_grad[i, j] = cost_result[:derivative]
              end
            end
          else
            arr = expected_output.as(Array)
            output_grad = SimpleMatrix.new(1, arr.size)
            arr.size.times do |i|
              expected = arr[i].as(GenNum).to_f64
              actual = actual_matrix[0, i]
              cost_result = cost_proc.call(expected, actual)
              output_grad[0, i] = cost_result[:derivative]
            end
          end

          output_grad = GPUMemory.to_gpu(output_grad)
          grad = output_layer.backward(output_grad)

          @hidden_layers.reverse_each do |layer|
            if layer.is_a?(MatrixLayer)
              grad = layer.backward(grad)
            elsif layer.is_a?(EmbeddingLayer)
              layer.accumulate_gradient
            end
          end
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
      update_lstm_gates(training_type) if @lstm_layers.any?

      batch_error
    end

    private def to_matrix(obj)
      if obj.is_a?(SimpleMatrix)
        GPUMemory.keep_on_gpu(obj.as(SimpleMatrix))
      else
        arr = obj.as(Array)
        if arr.size > 0 && arr[0].is_a?(Array)
          rows = arr.size
          cols = arr[0].as(Array).size
          mat = SimpleMatrix.new(rows, cols)
          rows.times do |i|
            cols.times do |j|
              mat[i, j] = arr[i].as(Array)[j].as(GenNum).to_f64
            end
          end
          GPUMemory.to_gpu(mat)
        else
          mat = SimpleMatrix.new(1, arr.size)
          arr.size.times do |i|
            mat[0, i] = arr[i].as(GenNum).to_f64
          end
          GPUMemory.to_gpu(mat)
        end
      end
    end

    def current_learning_rate
      if @warmup_steps > 0 && @time_step < @warmup_steps
        @learning_rate * (@time_step.to_f64 / @warmup_steps)
      else
        @learning_rate
      end
    end

    def update_lstm_gates(learn_type : Symbol | String)
      @lstm_layers.each do |layer|
        layer.update_gate_params(current_learning_rate)
      end
    end

    def update_transformer_layers
      lr = current_learning_rate
      @transformer_layers.each do |layer|
        layer.apply_gradients(lr)
      end
    end

    # Legacy neuron/synapse based weight update methods have been removed
    # in favor of the matrix-based implementation in the train method
    # and the layer-specific update_weights methods.

    def verify_data(data : Array(Array), label_mode : Bool = false)
      message = nil
      if data.sample.size != 2
        message = "Train data must have two arrays, one for input one for output"
      end
      sample_input = data.sample.first
      return if sample_input.is_a?(SimpleMatrix)
      if sample_input.is_a?(Array) && sample_input.as(Array).first.is_a?(Array)
        return
      end
      sample_input_arr = sample_input.is_a?(Array) ? sample_input.as(Array) : [sample_input]
      random_input = sample_input_arr.size
      sample_out = data.sample.last
      random_output = sample_out.is_a?(SimpleMatrix) ? sample_out.as(SimpleMatrix).cols : sample_out.as(Array).size
      data.each_with_index do |test, i|
        inp = test.first
        if !inp.is_a?(SimpleMatrix) && inp.as(Array).size != random_input
          message = "Input data sizes are inconsistent"
        end
        out = test.last
        out_size = out.is_a?(SimpleMatrix) ? out.as(SimpleMatrix).cols : out.size
        if out_size != random_output
          message = "Output data sizes are inconsistent"
        end
        unless label_mode || (out_size == @output_layers.first.size)
          message = "data at index #{i} and size: #{out_size} mismatch output layer size"
        end
      end
      if message
        Log.error { "#{message}: #{data}" }
        raise NeuralNetTrainError.new(message)
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

    # Helper method to safely multiply matrix by weights transpose, handling CUDA issues
    # and dimension mismatches
    private def safe_output_transform(matrix : SimpleMatrix, weights : SimpleMatrix) : SimpleMatrix
      begin
        # First check dimensions to provide a clearer error
        if matrix.cols != weights.cols
          raise ArgumentError.new("Matrix dimension mismatch: input features (#{matrix.cols}) doesn't match weights (#{weights.cols})")
        end

        w_t = weights.transpose
        matrix * w_t
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

        # Fallback to CPU computation if CUDA transpose fails
        if matrix.is_a?(CudaMatrix) && weights.is_a?(CudaMatrix)
          matrix_cpu = SimpleMatrix.new(matrix.rows, matrix.cols)
          matrix.rows.times { |i| matrix.cols.times { |j| matrix_cpu[i, j] = matrix[i, j] } }
          w_cpu = SimpleMatrix.new(weights.rows, weights.cols)
          weights.rows.times { |i| weights.cols.times { |j| w_cpu[i, j] = weights[i, j] } }
          result = matrix_cpu * w_cpu.transpose
          GPUMemory.to_gpu(result)
        else
          raise ex
        end
      end
    end
  end
end
