require "log"
require "json"
require "../cuda"
require "../math/simple_matrix"
require "../math/cuda_matrix"

module SHAInet
  class Network
    # ------------
    # There are no matrices in this implementation, instead the gradient values
    # are stored in each neuron/synapse independently.
    # When preforming propogation,
    # all the math is done iteratively on each neuron/synapse locally.

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
        matrix = out_layer.forward_matrix(matrix)

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
          else
            matrix = l.forward_matrix(matrix)
          end
        end
        out_layer = @output_layers.last
        matrix = out_layer.forward_matrix(matrix)
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
              # Update internal state matrices for matrix-based layers
              if out_layer.responds_to?(:neurons)
                out_layer.neurons[j].activation = act
                out_layer.neurons[j].sigma_prime = sig
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
            matrix = l.forward_matrix(matrix)
          end
        end
        out_layer = @output_layers.last
        matrix = out_layer.forward_matrix(matrix)
        matrix
      else
        raise NeuralNetRunError.new("Matrix input not supported for recurrent networks")
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # Run a batch of sequences by calling `run` for each sequence
    def run(input : Array(Array(Array(GenNum))), stealth : Bool = false) : Array(Array(Array(Float64)))
      input.map { |seq| run(seq, stealth: stealth) }
    end

    # Accept a sequence of integer tokens for embedding layers
    def run(input : Array(Array(Int32)), stealth : Bool = false) : Array(Array(Float64))
      seq = input.map { |x| x.map(&.to_f64) }
      run(seq, stealth: stealth)
    end

    # Accept integer input for embedding layers
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
            matrix = l.forward_matrix(matrix)
          end
        end
        out_layer = @output_layers.last
        matrix = out_layer.forward_matrix(matrix)
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

      # Get the error signal for the final layer, based on the cost function (error gradient is stored in the output neurons)
      @error_signal = [] of Float64 # Collect all the errors for current run

      actual_output.size.times do |i|
        cost = cost_function.call(expected_output[i], actual_output[i])
        @error_signal << cost[:value]

        # puts "Actual output: #{actual_output}"
        # puts "Cost value: #{cost[:value]}"
        # puts "Cost derivative: #{cost[:derivative]}"
        # puts "Neuron.sigma_prime: #{neuron.sigma_prime}"
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

      # Get the error signal for the final layer, based on the cost function (error gradient is stored in the output neurons)
      @error_signal = [] of Float64 # Collect all the errors for current run

      actual_output.size.times do |i|
        cost = cost_function.call(expected_output[i], actual_output[i])
        @error_signal << cost[:value]

        # puts "Actual output: #{actual_output}"
        # puts "Cost value: #{cost[:value]}"
        # puts "Cost derivative: #{cost[:derivative]}"
        # puts "Neuron.sigma_prime: #{neuron.sigma_prime}"
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

    def evaluate_sequence(input_data : Array(Array(Int32)),
                          expected_output : Array(GenNum),
                          cost_function : CostFunction = SHAInet.quadratic_cost)
      seq = input_data.map { |x| x.map(&.to_f64) }
      evaluate_sequence(seq, expected_output, cost_function)
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

    def evaluate_label(input_data : Array(Int32), label : Int32)
      evaluate_label(input_data.map(&.to_f64), label)
    end

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

    # Training the model
    # ameba:disable Metrics/CyclomaticComplexity
    def train(data : Array(Array) | SHAInet::TrainingData | SHAInet::StreamingData, # Input may contain sequences
              training_type : Symbol | String,                                      # Type of training: :sgdm, :rprop, :adam
              cost_function : Symbol | String | CostFunction = :mse,                # Proc returns the function value and it's derivative
              epochs : Int32 = 1,                                                   # a criteria of when to stop the training
              error_threshold : Float64 = 0.00000001,                               # a criteria of when to stop the training
              mini_batch_size : Int32 = 1,                                          # Size of mini-batches to train with
              log_each : Int32 = 1000,                                              # determines what is the step for error printout
              show_slice : Bool = false,                                            # Show progress of each mini-batch slice
              autosave : NamedTuple(freq: Int32, path: String) | Nil = nil)         # Save the network each X epochs

      # This methods accepts data as either a SHAInet::TrainingData object, an Array(Array(Array(GenNum))),
      # or a SHAInet::StreamingData instance.
      raw_data = nil
      use_gpu = CUDA.available?
      if data.is_a?(SHAInet::StreamingData)
        Log.info { "Training started" }
        Log.info { "CUDA available: #{use_gpu}" }
        start_time = Time.monotonic
        batch_size = mini_batch_size
        @time_step = 0
        if CUDA.available?
          input_dim = @input_layers.reduce(0) { |acc, l| acc + l.size }
          out_dim = @output_layers.last.size
          GPUMemory.preallocate!(1, input_dim, batch_size)
          GPUMemory.preallocate!(1, out_dim, batch_size)
        end
      else
        raw_data = data.is_a?(SHAInet::TrainingData) ? data.data : data
        Log.info { "Training started" }
        Log.info { "CUDA available: #{use_gpu}" }
        start_time = Time.monotonic
        batch_size = mini_batch_size ? mini_batch_size : raw_data.size
        @time_step = 0
        if CUDA.available?
          input_dim = @input_layers.reduce(0) { |acc, l| acc + l.size }
          out_dim = @output_layers.last.size
          GPUMemory.preallocate!(1, input_dim, batch_size)
          GPUMemory.preallocate!(1, out_dim, batch_size)
        end
      end

      # Change String/Symbol into the corrent proc of the cost function
      label_mode = false
      if cost_function.is_a?(Symbol) || cost_function.is_a?(String)
        raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :c_ent_sm, :exp, :hel_d, :kld, :gkld, :ita_sai_d).") if COST_FUNCTIONS.any? { |x| x == cost_function.to_s } == false
        label_mode = cost_function.to_s == "c_ent_sm"
        unless label_mode
          proc = get_cost_proc(cost_function.to_s)
          cost_function = proc
        end
      end

      counter = 0_i64
      loop do
        # Autosave the network
        unless autosave.nil?
          if counter % autosave[:freq] == 0 && (counter > 0)
            # Log.info { "Network saved." }
            save_to_file("#{autosave[:path]}/autosave_epoch_#{counter}.nn")
          end
        end

        # Break condtitions
        if counter >= epochs || (error_threshold >= @mse) && (counter > 1)
          log_summary(counter)
          Log.info { "Training finished. (Elapsed: #{Time.monotonic - start_time})" }
          break
        end

        # Show training progress of epochs
        if counter % log_each == 0
          log_summary(counter)
          # @all_neurons.each { |s| puts s.gradient }
        end

        # Counters for disply
        display_counter = 0
        slices = 0

        # For error break condition
        epoch_mse = 0.0
        epoch_error_sum = Array(Float64).new(@output_layers.last.size) { 0.0 }

        # Iterate over each mini-batch
        if data.is_a?(SHAInet::StreamingData)
          acc_counter = 0
          while (data_slice = data.next_batch(batch_size)).size > 0
            slices += 1
            verify_data(data_slice, label_mode)
            @time_step += 1 if mini_batch_size
            batch_mean = 0.0_f64
            all_errors = 0.0_f64
            if acc_counter % @accumulation_steps == 0
              @w_gradient.fill(0.0)
              @b_gradient.fill(0.0)
              @lstm_layers.each &.zero_gate_gradients
              @transformer_layers.each &.zero_gradients
            end
            data_slice.each do |data_point|
              input_d = data_point[0]
              output_d = data_point[1]

              if input_d.is_a?(SimpleMatrix)
                if label_mode
                  label = output_d.as(SimpleMatrix)[0, 0].to_i
                  evaluate_label(input_d.as(SimpleMatrix), label)
                else
                  evaluate(input_d.as(SimpleMatrix), output_d.as(SimpleMatrix), cost_function.as(CostFunction))
                end
              else
                output_arr = [] of Float64
                if output_d.is_a?(Array)
                  output_d.as(Array).each do |v|
                    if v.is_a?(Number)
                      output_arr << v.to_f64
                    end
                  end
                elsif output_d.is_a?(SimpleMatrix)
                  output_arr << output_d.as(SimpleMatrix)[0, 0]
                else
                  output_arr << output_d.to_f64
                end

                if label_mode
                  label = output_arr.first.to_i
                  if input_d.is_a?(Array) && input_d.as(Array).first.is_a?(Array)
                    seq_in = (input_d.as(Array).map { |x| x.as(Array).map(&.to_f64).as(Array(Float64)) }).as(Array(Array(Float64)))
                    evaluate_sequence_label(seq_in, label)
                  else
                    input_arr = [] of Float64
                    input_d.as(Array).each do |v|
                      input_arr << v.to_f64 if v.is_a?(Number)
                    end
                    evaluate_label(input_arr, label)
                  end
                else
                  if input_d.is_a?(Array) && input_d.as(Array).first.is_a?(Array)
                    seq_in = (input_d.as(Array).map { |x| x.as(Array).map(&.to_f64).as(Array(Float64)) }).as(Array(Array(Float64)))
                    evaluate_sequence(seq_in, output_arr.map(&.to_f64), cost_function.as(CostFunction))
                  else
                    input_arr = [] of Float64
                    input_d.as(Array).each do |v|
                      input_arr << v.to_f64 if v.is_a?(Number)
                    end
                    evaluate(input_arr, output_arr.map(&.to_f64), cost_function.as(CostFunction))
                  end
                end
              end
              all_errors += @total_error
              if @hidden_layers.any? &.is_a?(TransformerLayer)
                grad = GPUMemory.keep_on_gpu(@transformer_error)
                prev = @output_layers.last
                @hidden_layers.reverse_each do |l|
                  if l.is_a?(TransformerLayer)
                    grad = l.as(TransformerLayer).backward(grad)
                  else
                    grad = l.backward_matrix(prev, grad)
                  end
                  prev = l
                end
                @input_layers.reverse_each do |l|
                  grad = l.backward_matrix(prev, grad)
                  prev = l
                end
              elsif @hidden_layers.none? { |l| l.is_a?(RecurrentLayer) || l.is_a?(LSTMLayer) }
                # For matrix-based layers, use matrix-based gradients
                grad = @output_layers.last.activations.clone
                prev = @output_layers.last
                @hidden_layers.reverse_each do |l|
                  grad = l.backward_matrix(prev, grad)
                  prev = l
                end
                @input_layers.reverse_each do |l|
                  grad = l.backward_matrix(prev, grad)
                  prev = l
                end
              else
                # Old neuron-based logic removed in favor of matrix operations
                # For non-transformer, non-LSTM layers, use matrix backpropagation
                @hidden_layers.reverse_each do |l|
                  if l.responds_to?(:neurons)
                    l.neurons.each { |neuron| neuron.hidden_error_prop }
                  end
                end
                @input_layers.reverse_each do |l|
                  if l.responds_to?(:neurons)
                    l.neurons.each { |neuron| neuron.hidden_error_prop }
                  end
                end
              end
              @hidden_layers.each do |l|
                if l.is_a?(EmbeddingLayer)
                  l.as(EmbeddingLayer).accumulate_gradient
                end
              end
              # Gradient collection for matrix-based layers moved to layer-level operations
              # @all_synapses.each_with_index { |synapse, i| @w_gradient[i] += (synapse.source_neuron.activation)*(synapse.dest_neuron.gradient) }
              # @all_neurons.each_with_index { |neuron, i| @b_gradient[i] += neuron.gradient }
              @lstm_layers.each &.accumulate_gate_gradients
              if @error_signal.size == 1
                error_avg = 0.0_f64
              else
                # Calculate error based on output layer size
                output_size = @output_layers.last.size
                error_avg = @total_error / output_size
              end
              sqrd_dists = 0.0_f64
              @error_signal.each { |e| sqrd_dists += (e - error_avg)**2 }
              @mse = sqrd_dists / @output_layers.last.size
              batch_mean += @mse
            end

            @total_error = all_errors
            batch_mean /= data_slice.size
            @mse = batch_mean
            acc_counter += 1
            if acc_counter % @accumulation_steps == 0
              @time_step += 1 unless mini_batch_size
              update_lstm_gates(training_type)
              update_transformer_layers
              epoch_mse += @mse
              @error_signal.size.times { |i| epoch_error_sum[i] += @error_signal[i] }
              @prev_mse = @mse.clone
            end
            display_counter += 1
            if counter % log_each == 0
              Log.info { "  Slice: (#{display_counter}), MSE: #{@mse}" } if show_slice
            end
          end
          # end streaming while loop
          data.rewind
        else
          acc_counter = 0
          raw_data.not_nil!.each_slice(batch_size, reuse: false) do |data_slice|
            slices += 1
            verify_data(data_slice, label_mode)
            @time_step += 1 if mini_batch_size # in mini-batch update adam time_step

            batch_mean = 0.0_f64
            all_errors = 0.0_f64

            if acc_counter % @accumulation_steps == 0
              @w_gradient.fill(0.0)
              @b_gradient.fill(0.0)
              @lstm_layers.each &.zero_gate_gradients
              @transformer_layers.each &.zero_gradients
            end

            # Go over each data point and collect gradients of weights/biases
            # based on each specific example
            data_slice.each do |data_point|
              input_d = data_point[0]
              output_d = data_point[1]
              output_arr = [] of Float64
              if output_d.is_a?(Array)
                output_d.as(Array).each do |v|
                  if v.is_a?(Number)
                    output_arr << v.to_f64
                  end
                end
              else
                output_arr << output_d.to_f64
              end
              if label_mode
                label = output_arr.first.to_i
                if input_d.is_a?(Array) && input_d.as(Array).first.is_a?(Array)
                  seq_in = (input_d.as(Array).map { |x| x.as(Array).map(&.to_f64).as(Array(Float64)) }).as(Array(Array(Float64)))
                  evaluate_sequence_label(seq_in, label)
                else
                  input_arr = [] of Float64
                  input_d.as(Array).each do |v|
                    input_arr << v.to_f64 if v.is_a?(Number)
                  end
                  evaluate_label(input_arr, label)
                end
              else
                if input_d.is_a?(Array) && input_d.as(Array).first.is_a?(Array)
                  seq_in = (input_d.as(Array).map { |x| x.as(Array).map(&.to_f64).as(Array(Float64)) }).as(Array(Array(Float64)))
                  evaluate_sequence(seq_in, output_arr.map(&.to_f64), cost_function.as(CostFunction))
                else
                  input_arr = [] of Float64
                  input_d.as(Array).each do |v|
                    input_arr << v.to_f64 if v.is_a?(Number)
                  end
                  evaluate(input_arr, output_arr.map(&.to_f64), cost_function.as(CostFunction))
                end
              end
              # all_errors << @total_error
              all_errors += @total_error

              if @hidden_layers.any? &.is_a?(TransformerLayer)
                @hidden_layers.reverse_each do |l|
                  if l.is_a?(TransformerLayer)
                    l.as(TransformerLayer).backward(@transformer_error)
                  end
                end
              elsif @hidden_layers.none? { |l| l.is_a?(RecurrentLayer) || l.is_a?(LSTMLayer) }
                grad = nil
                prev = @output_layers.last
                @hidden_layers.reverse_each do |l|
                  grad = l.backward_matrix(prev, grad)
                  prev = l
                end
                @input_layers.reverse_each do |l|
                  grad = l.backward_matrix(prev, grad)
                  prev = l
                end
              else
                # For non-matrix layers that might still have neurons
                @hidden_layers.reverse_each do |l|
                  if l.responds_to?(:neurons)
                    l.neurons.each { |neuron| neuron.hidden_error_prop }
                  end
                end
                @input_layers.reverse_each do |l|
                  if l.responds_to?(:neurons)
                    l.neurons.each { |neuron| neuron.hidden_error_prop }
                  end
                end
              end

              # Collect gradients for embedding layers before summing
              @hidden_layers.each do |l|
                if l.is_a?(EmbeddingLayer)
                  l.as(EmbeddingLayer).accumulate_gradient
                end
              end

              # Matrix-based gradient accumulation handled by layer-level operations
              # @all_synapses.each_with_index { |synapse, i| @w_gradient[i] += (synapse.source_neuron.activation)*(synapse.dest_neuron.gradient) }
              # @all_neurons.each_with_index { |neuron, i| @b_gradient[i] += neuron.gradient }
              @lstm_layers.each &.accumulate_gate_gradients

              # Calculate MSE per data point
              if @error_signal.size == 1
                error_avg = 0.0_f64
              else
                error_avg = @total_error/@output_layers.last.size
              end
              # sqrd_dists = [] of Float64
              # @error_signal.each { |e| sqrd_dists << (e - error_avg)**2 }
              sqrd_dists = 0.0_f64
              @error_signal.each { |e| sqrd_dists += (e - error_avg)**2 }

              # @mse = (sqrd_dists.reduce { |acc, i| acc + i })/@output_layers.last.size
              # batch_mean << @mse
              @mse = sqrd_dists / @output_layers.last.size
              batch_mean += @mse
            end

            # Total error per batch
            # @total_error = all_errors.reduce { |acc, i| acc + i }
            @total_error = all_errors

            # Calculate MSE per batch
            # batch_mean = (batch_mean.reduce { |acc, i| acc + i })/data_slice.size
            # @mse = batch_mean
            batch_mean /= data_slice.size
            @mse = batch_mean

            acc_counter += 1
            if acc_counter % @accumulation_steps == 0
              @time_step += 1 unless mini_batch_size
              update_weights(training_type)
              update_biases(training_type)
              update_lstm_gates(training_type)
              update_transformer_layers

              epoch_mse += @mse
              @error_signal.size.times { |i| epoch_error_sum[i] += @error_signal[i] }

              @prev_mse = @mse.clone
            end
            # Show training progress of the mini-batches
            display_counter += 1
            if counter % log_each == 0
              Log.info { "  Slice: (#{display_counter} / #{slices}), MSE: #{@mse}" } if show_slice
              # Log.info { "@error_signal: #{@error_signal}" }
            end
          end
        end

        # Update epoch status
        @mse = (epoch_mse / slices)
        @error_signal.size.times { |i| @error_signal[i] = (epoch_error_sum[i] / slices) }
        counter += 1
      end
    ensure
      GPUMemory.cleanup if CUDA.available?
    end

    # This method is kept for matching syntax of previous versions.
    # It is possible to use the "train" method instead
    def train_batch(data : Array(Array) | SHAInet::TrainingData,
                    training_type : Symbol | String = :sgdm,
                    cost_function : Symbol | String | CostFunction = :mse,
                    epochs : Int32 = 1,
                    error_threshold : Float64 = 0.00000001,
                    mini_batch_size : Int32 = 1,
                    log_each : Int32 = 1,
                    show_slice : Bool = false,
                    autosave : NamedTuple(freq: Int32, path: String) | Nil = nil)
      train(data: data,
        training_type: training_type,
        cost_function: cost_function,
        epochs: epochs,
        error_threshold: error_threshold,
        mini_batch_size: mini_batch_size,
        log_each: log_each,
        show_slice: show_slice,
        autosave: autosave)
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

    # Use evolutionary strategies for network optimization instread of gradient based approach
    # ameba:disable Metrics/CyclomaticComplexity
    def train_es(data : Array(Array(Array(GenNum))) | SHAInet::TrainingData,   # Input structure: data = [[Input = [] of Float64],[Expected result = [] of Float64]]
                 pool_size : Int32,                                            # How many random direction to try each time
                 learning_rate : Float64,                                      # How much of the noise i used for the parameter update
                 sigma : Float64,                                              # Range of noise values
                 cost_function : Symbol | String | CostFunction = :c_ent,      # Proc returns the function value and it's derivative
                 epochs : Int32 = 1,                                           # a criteria of when to stop the training
                 mini_batch_size : Int32 = 1,                                  # Size of batch
                 error_threshold : Float64 = 0.0,                              # a criteria of when to stop the training
                 log_each : Int32 = 1,                                         # determines what is the step for error printout
                 show_slice : Bool = false,                                    # Show MSE for each slice
                 autosave : NamedTuple(freq: Int32, path: String) | Nil = nil) # Save the network each X epochs)
      # This methods accepts data as either a SHAInet::TrainingData object, or as an Array(Array(Array(GenNum)).
      # In the case of SHAInet::TrainingData, we convert it to an Array(Array(Array(GenNum)) by calling #data on it.
      raw_data = data.is_a?(SHAInet::TrainingData) ? data.data : data
      Log.info { "Training started" }
      start_time = Time.monotonic
      batch_size = mini_batch_size ? mini_batch_size : raw_data.size

      # Change String/Symbol into the corrent proc of the cost function
      if cost_function.is_a?(Symbol) || cost_function.is_a?(String)
        raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :c_ent_sm, :exp, :hel_d, :kld, :gkld, :ita_sai_d).") if COST_FUNCTIONS.any? { |x| x == cost_function.to_s } == false
        proc = get_cost_proc(cost_function.to_s)
        cost_function = proc
      end

      epoch = 0_i64
      loop do
        # Autosave the network
        unless autosave.nil?
          if epoch % autosave[:freq] == 0 && (epoch > 0)
            # Log.info { "Network saved." }
            save_to_file("#{autosave[:path]}/autosave_epoch_#{epoch}.nn")
          end
        end

        # Break condtitions
        if epoch >= epochs || (error_threshold >= @mse) && (epoch > 1)
          log_summary(epoch)
          Log.info { "Training finished. (Elapsed: #{Time.monotonic - start_time})" }
          break
        end

        # Show training progress of epochs
        if epoch % log_each == 0
          log_summary(epoch)
          # @all_neurons.each { |s| puts s.gradient }
        end

        # Counters for disply
        display_counter = 0
        slices = (data.size.to_f64 / mini_batch_size).ceil.to_i

        # For error break condition
        epoch_mse = 0.0
        epoch_error_sum = Array(Float64).new(@output_layers.last.size) { 0.0 }

        raw_data.not_nil!.each_slice(batch_size, reuse: false) do |data_slice|
          verify_data(data_slice, false)

          pool = Pool.new(
            network: self,
            pool_size: pool_size,
            learning_rate: learning_rate,
            sigma: sigma)

          # Update wieghts & biases for the batch
          pool.organisms.each do |organism|
            organism.get_new_params # Get new weights & biases

            # Go over each data points and collect errors
            # based on each specific example in the batch
            batch_mse_sum = 0.0_f64
            batch_errors_sum = Array(Float64).new(@output_layers.last.size) { 0.0 }

            data_slice.each do |data_point|
              # Update error signal in output layer
              input_d = data_point[0]
              output_d = data_point[1]
              output_arr = [] of Float64
              if output_d.is_a?(Array)
                output_d.as(Array).each do |v|
                  if v.is_a?(Number)
                    output_arr << v.to_f64
                  end
                end
              else
                output_arr << output_d.to_f64
              end
              if input_d.is_a?(Array) && input_d.as(Array).first.is_a?(Array)
                seq_in = (input_d.as(Array).map { |x| x.as(Array).map(&.to_f64).as(Array(Float64)) }).as(Array(Array(Float64)))
                evaluate_sequence(seq_in, output_arr.map(&.to_f64), cost_function.as(CostFunction))
              else
                input_arr = [] of Float64
                input_d.as(Array).each do |v|
                  if v.is_a?(Number)
                    input_arr << v.to_f64
                  end
                end
                evaluate(input_arr, output_arr.map(&.to_f64), cost_function.as(CostFunction))
              end
              update_mse
              batch_mse_sum += @mse
              @error_signal.size.times { |i| batch_errors_sum[i] += @error_signal[i] }
            end

            @mse = (batch_mse_sum / mini_batch_size) # Update MSE of the batch
            @error_signal = batch_errors_sum
            organism.update_reward
          end

          epoch_mse += @mse
          @error_signal.size.times { |i| epoch_error_sum[i] += @error_signal[i] }
          pool.pull_params

          # Show training progress of the mini-batches
          display_counter += 1
          if epoch % log_each == 0
            Log.info { "  Slice: (#{display_counter} / #{slices}), MSE: #{@mse}" } if show_slice
            # Log.info { "@error_signal: #{@error_signal}" }
          end
        end
        # Update epoch status
        @mse = (epoch_mse / slices)
        @error_signal.size.times { |i| @error_signal[i] = (epoch_error_sum[i] / slices) }
        epoch += 1
      end
    end

    # Update weights based on gradient information
    def update_weights(training_type)
      lr = current_learning_rate

      # Handle all layer types
      @all_layers.each do |layer|
        case layer
        when EmbeddingLayer
          layer.as(EmbeddingLayer).accumulate_gradient
          layer.as(EmbeddingLayer).apply_gradients(lr)
        when LSTMLayer
          layer.as(LSTMLayer).update_gate_params(lr)
        when TransformerLayer
          layer.as(TransformerLayer).apply_gradients(lr)
        else
          # Standard matrix-based layers
          # No explicit update needed as gradients are applied during backprop
        end
      end

      # Handle transformer layers separately if they exist
      update_transformer_layers if @transformer_layers.any?
    end

    # Update biases based on gradient information
    def update_biases(training_type)
      # Matrix-based implementation - biases are updated during backprop
      # This method is kept for backward compatibility
    end

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
