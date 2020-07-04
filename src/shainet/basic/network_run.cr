require "log"
require "json"

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
      raise NeuralNetRunError.new(
        "Error input data size: #{input.size} doesn't fit input layer size: #{@input_layers.first.neurons.size}.") unless input.size == @input_layers.first.neurons.size

      # Insert the input data into the input layer
      input.each_with_index do |data, i|
        # Inserts the input information into the input layers
        # TODO: add support for multiple input layers
        @input_layers.first.neurons[i].activation = data.to_f64
      end

      # Propogate the information forward through the hidden layers

      @hidden_layers.each do |l|
        l.neurons.each { |neuron| neuron.activate(l.activation_function) }
      end

      # Propogate the information through the output layers
      @output_layers.each do |l|
        l.neurons.each { |neuron| neuron.activate(l.activation_function) }
      end

      output = @output_layers.last.neurons.map { |neuron| neuron.activation } # return an array of all output neuron activations
      # TODO: add support for multiple output layers

      unless stealth # Hide output report during training
        @log.info { "Input => #{input}, network output => #{output}" }
      end
      output
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # Quantifies how good the network performed for a single input compared to the expected output
    # This function returns the actual output and updates the error gradient for the output layer
    def evaluate(input_data : Array(GenNum),
                 expected_output : Array(GenNum),
                 cost_function : CostFunction = SHAInet.quadratic_cost)
      #
      actual_output = run(input_data, stealth: true)

      # Test for NaNs & exploading gradients
      validate_values(actual_output, "actual_output")

      # Get the error signal for the final layer, based on the cost function (error gradient is stored in the output neurons)
      @error_signal = [] of Float64 # Collect all the errors for current run

      actual_output.size.times do |i|
        neuron = @output_layers.last.neurons[i] # Update error of all neurons in the output layer based on the actual result
        cost = cost_function.call(expected_output[i], actual_output[i])
        neuron.gradient = cost[:derivative]*neuron.sigma_prime
        @error_signal << cost[:value] # Store the output error based on cost function

        # puts "Actual output: #{actual_output}"
        # puts "Cost value: #{cost[:value]}"
        # puts "Cost derivative: #{cost[:derivative]}"
        # puts "Neuron.sigma_prime: #{neuron.sigma_prime}"
        # puts "---"
      end

      # Test for NaNs & exploading gradients
      validate_values(@error_signal, "error_signal")
      @total_error = @error_signal.reduce(0.0) { |acc, i| acc + i } # Sum up all the errors from output layer

      # puts "@error_signal: #{@error_signal}"
      # puts "@total_error: #{@total_error}"


    rescue e : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{e}")
    end

    # Calculate MSE from the error signal of the output layer
    def update_mse
      n = @output_layers.last.neurons.size
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
    def train(data : Array(Array(Array(GenNum))) | SHAInet::TrainingData,   # Input structure: data = [[Input = [] of Float64],[Expected result = [] of Float64]]
              training_type : Symbol | String,                              # Type of training: :sgdm, :rprop, :adam
              cost_function : Symbol | String | CostFunction = :mse,        # Proc returns the function value and it's derivative
              epochs : Int32 = 1,                                           # a criteria of when to stop the training
              error_threshold : Float64 = 0.00000001,                       # a criteria of when to stop the training
              mini_batch_size : Int32 = 1,                                  # Size of mini-batches to train with
              log_each : Int32 = 1000,                                      # determines what is the step for error printout
              show_slice : Bool = false,                                    # Show progress of each mini-batch slice
              autosave : NamedTuple(freq: Int32, path: String) | Nil = nil) # Save the network each X epochs

      # This methods accepts data as either a SHAInet::TrainingData object, or as an Array(Array(Array(GenNum)).
      # In the case of SHAInet::TrainingData, we convert it to an Array(Array(Array(GenNum)) by calling #data on it.
      raw_data = data.is_a?(SHAInet::TrainingData) ? data.data : data
      @log.info { "Training started" }
      start_time = Time.local
      batch_size = mini_batch_size ? mini_batch_size : raw_data.size
      @time_step = 0

      # Change String/Symbol into the corrent proc of the cost function
      if cost_function.is_a?(Symbol) || cost_function.is_a?(String)
        raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d).") if COST_FUNCTIONS.any? { |x| x == cost_function.to_s } == false
        proc = get_cost_proc(cost_function.to_s)
        cost_function = proc
      end

      counter = 0_i64
      loop do
        # Autosave the network
        unless autosave.nil?
          if counter % autosave[:freq] == 0 && (counter > 0)
            # @log.info("Network saved.")
            save_to_file("#{autosave[:path]}/autosave_epoch_#{counter}.nn")
          end
        end

        # Break condtitions
        if counter >= epochs || (error_threshold >= @mse) && (counter > 1)
          log_summary(counter)
          @log.info { "Training finished. (Elapsed: #{Time.local - start_time})" }
          break
        end

        # Show training progress of epochs
        if counter % log_each == 0
          log_summary(counter)
          # @all_neurons.each { |s| puts s.gradient }
        end

        # Counters for disply
        display_counter = 0
        slices = (data.size.to_f64 / mini_batch_size).ceil.to_i

        # For error break condition
        epoch_mse = 0.0
        epoch_error_sum = Array(Float64).new(@output_layers.last.neurons.size) { 0.0 }

        # Iterate over each mini-batch
        raw_data.each_slice(batch_size, reuse: false) do |data_slice|
          verify_data(data_slice)
          @time_step += 1 if mini_batch_size # in mini-batch update adam time_step

          # batch_mean = [] of Float64
          # all_errors = [] of Float64
          batch_mean = 0.0_f64
          all_errors = 0.0_f64

          # Save gradients from entire batch before updating weights & biases
          @w_gradient = Array(Float64).new(@all_synapses.size) { 0.0 }
          @b_gradient = Array(Float64).new(@all_neurons.size) { 0.0 }

          # Go over each data point and collect gradients of weights/biases
          # based on each specific example
          data_slice.each do |data_point|
            evaluate(data_point[0], data_point[1], cost_function) # Get error gradient from output layer based on current input
            # all_errors << @total_error
            all_errors += @total_error

            # Propogate the errors backwards through the hidden layers
            @hidden_layers.reverse_each do |l|
              # Update neuron error based on errors*weights of neurons from the next layer
              l.neurons.each { |neuron| neuron.hidden_error_prop }
            end

            # Propogate the errors backwards through the input layers
            @input_layers.reverse_each do |l|
              # Update neuron error based on errors*weights of neurons from the next layer
              l.neurons.each { |neuron| neuron.hidden_error_prop }
            end

            # Sum all gradients from each data point for the batch update
            @all_synapses.each_with_index { |synapse, i| @w_gradient[i] += (synapse.source_neuron.activation)*(synapse.dest_neuron.gradient) }
            @all_neurons.each_with_index { |neuron, i| @b_gradient[i] += neuron.gradient }

            # Calculate MSE per data point
            if @error_signal.size == 1
              error_avg = 0.0_f64
            else
              error_avg = @total_error/@output_layers.last.neurons.size
            end
            # sqrd_dists = [] of Float64
            # @error_signal.each { |e| sqrd_dists << (e - error_avg)**2 }
            sqrd_dists = 0.0_f64
            @error_signal.each { |e| sqrd_dists += (e - error_avg)**2 }

            # @mse = (sqrd_dists.reduce { |acc, i| acc + i })/@output_layers.last.neurons.size
            # batch_mean << @mse
            @mse = sqrd_dists / @output_layers.last.neurons.size
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

          # Update all wieghts & biases for the batch
          @time_step += 1 unless mini_batch_size # Based on how many epochs have passed in current training run, needed for Adam
          update_weights(training_type)
          update_biases(training_type)

          # Update epoch status
          epoch_mse += @mse
          @error_signal.size.times { |i| epoch_error_sum[i] += @error_signal[i] }

          @prev_mse = @mse.clone
          # Show training progress of the mini-batches
          display_counter += 1
          if counter % log_each == 0
            @log.info { "  Slice: (#{display_counter} / #{slices}), MSE: #{@mse}" } if show_slice
            # @log.info("@error_signal: #{@error_signal}")
          end
        end

        # Update epoch status
        @mse = (epoch_mse / slices)
        @error_signal.size.times { |i| @error_signal[i] = (epoch_error_sum[i] / slices) }
        counter += 1
      end
    end

    # This method is kept for matching syntax of previous versions.
    # It is possible to use the "train" method instead
    def train_batch(data : Array(Array(Array(GenNum))) | SHAInet::TrainingData,
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

    # Update weights based on the learning type chosen
    def update_weights(learn_type : Symbol | String, batch : Bool = false)
      @all_synapses.each_with_index do |synapse, i|
        # Get current gradient (needed for mini-batch)
        synapse.gradient = @w_gradient.not_nil![i]

        case learn_type.to_s
        # Update weights based on the gradients and delta rule (including momentum)
        when "sgdm"
          delta_weight = (-1)*@learning_rate*synapse.gradient + @momentum*(synapse.weight - synapse.prev_weight)
          synapse.weight += delta_weight
          synapse.prev_weight = synapse.weight

          # Update weights based on Resilient backpropogation (Rprop), using the improved varient iRprop+
        when "rprop"
          if synapse.prev_gradient*synapse.gradient > 0
            delta = [@etah_plus*synapse.prev_delta, @delta_max].min
            delta_weight = (-1)*SHAInet.sign(synapse.gradient)*delta

            synapse.weight += delta_weight
            synapse.prev_weight = synapse.weight
            synapse.prev_delta = delta
            synapse.prev_delta_w = delta_weight
          elsif synapse.prev_gradient*synapse.gradient < 0.0
            delta = [@etah_minus*synapse.prev_delta, @delta_min].max

            synapse.weight -= synapse.prev_delta_w if @mse >= @prev_mse

            synapse.prev_gradient = 0.0
            synapse.prev_delta = delta
          elsif synapse.prev_gradient*synapse.gradient == 0.0
            delta_weight = (-1)*SHAInet.sign(synapse.gradient)*synapse.prev_delta

            synapse.weight += delta_weight
            synapse.prev_delta = @delta_min
            synapse.prev_delta_w = delta_weight
          end
          # Update weights based on Adaptive moment estimation (Adam)
        when "adam"
          synapse.m_current = @beta1*synapse.m_prev + (1 - @beta1)*synapse.gradient
          synapse.v_current = @beta2*synapse.v_prev + (1 - @beta2)*(synapse.gradient)**2

          m_hat = synapse.m_current/(1 - @beta1**@time_step)
          v_hat = synapse.v_current/(1 - @beta2**@time_step)
          synapse.weight -= (@alpha*m_hat)/(v_hat**0.5 + @epsilon)

          synapse.m_prev = synapse.m_current
          synapse.v_prev = synapse.v_current
        end
      end
    end

    # Update biases based on the learning type chosen
    def update_biases(learn_type : Symbol | String, batch : Bool = false)
      @all_neurons.each_with_index do |neuron, i|
        # Get current gradient (needed for mini-batch)
        neuron.gradient = @b_gradient.not_nil![i]

        case learn_type.to_s
        # Update biases based on the gradients and delta rule (including momentum)
        when "sgdm"
          delta_bias = (-1)*@learning_rate*(neuron.gradient) + @momentum*(neuron.bias - neuron.prev_bias)
          neuron.bias += delta_bias
          neuron.prev_bias = neuron.bias

          # Update weights based on Resilient backpropogation (Rprop), using the improved varient iRprop+
        when "rprop"
          if neuron.prev_gradient*neuron.gradient > 0
            delta = [@etah_plus*neuron.prev_delta, @delta_max].min
            delta_bias = (-1)*SHAInet.sign(neuron.gradient)*delta

            neuron.bias += delta_bias
            neuron.prev_bias = neuron.bias
            neuron.prev_delta = delta
            neuron.prev_delta_b = delta_bias
          elsif neuron.prev_gradient*neuron.gradient < 0.0
            delta = [@etah_minus*neuron.prev_delta, @delta_min].max

            neuron.bias -= neuron.prev_delta_b if @mse >= @prev_mse

            neuron.prev_gradient = 0.0
            neuron.prev_delta = delta
          elsif neuron.prev_gradient*neuron.gradient == 0.0
            delta_bias = (-1)*SHAInet.sign(neuron.gradient)*@delta_min*neuron.prev_delta

            neuron.bias += delta_bias
            neuron.prev_delta = @delta_min
            neuron.prev_delta_b = delta_bias
          end
          # Update weights based on Adaptive moment estimation (Adam)
        when "adam"
          neuron.m_current = @beta1*neuron.m_prev + (1 - @beta1)*neuron.gradient
          neuron.v_current = @beta2*neuron.v_prev + (1 - @beta2)*(neuron.gradient)**2

          m_hat = neuron.m_current/(1 - @beta1**@time_step)
          v_hat = neuron.v_current/(1 - @beta2**@time_step)

          neuron.bias -= (@alpha*m_hat)/(v_hat**0.5 + @epsilon)

          neuron.m_prev = neuron.m_current
          neuron.v_prev = neuron.v_current
        end
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
      @log.info { "Training started" }
      start_time = Time.local
      batch_size = mini_batch_size ? mini_batch_size : raw_data.size

      # Change String/Symbol into the corrent proc of the cost function
      if cost_function.is_a?(Symbol) || cost_function.is_a?(String)
        raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d).") if COST_FUNCTIONS.any? { |x| x == cost_function.to_s } == false
        proc = get_cost_proc(cost_function.to_s)
        cost_function = proc
      end

      epoch = 0_i64
      loop do
        # Autosave the network
        unless autosave.nil?
          if epoch % autosave[:freq] == 0 && (epoch > 0)
            # @log.info("Network saved.")
            save_to_file("#{autosave[:path]}/autosave_epoch_#{epoch}.nn")
          end
        end

        # Break condtitions
        if epoch >= epochs || (error_threshold >= @mse) && (epoch > 1)
          log_summary(epoch)
          @log.info { "Training finished. (Elapsed: #{Time.local - start_time})" }
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
        epoch_error_sum = Array(Float64).new(@output_layers.last.neurons.size) { 0.0 }

        raw_data.each_slice(batch_size, reuse: false) do |data_slice|
          verify_data(data_slice)

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
            batch_errors_sum = Array(Float64).new(@output_layers.last.neurons.size) { 0.0 }

            data_slice.each do |data_point|
              # Update error signal in output layer
              evaluate(data_point[0], data_point[1], cost_function)
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
            @log.info { "  Slice: (#{display_counter} / #{slices}), MSE: #{@mse}" } if show_slice
            # @log.info("@error_signal: #{@error_signal}")
          end
        end
        # Update epoch status
        @mse = (epoch_mse / slices)
        @error_signal.size.times { |i| @error_signal[i] = (epoch_error_sum[i] / slices) }
        epoch += 1
      end
    end

    def verify_data(data : Array(Array(Array(GenNum))))
      message = nil
      if data.sample.size != 2
        message = "Train data must have two arrays, one for input one for output"
      end
      random_input = data.sample.first.size
      random_output = data.sample.last.size
      data.each_with_index do |test, i|
        if (test.first.size != random_input)
          message = "Input data sizes are inconsistent"
        end
        if (test.last.size != random_output)
          message = "Output data sizes are inconsistent"
        end
        unless (test.last.size == @output_layers.first.neurons.size)
          message = "data at index #{i} and size: #{test.last.size} mismatch output layer size"
        end
      end
      if message
        @log.error { "#{message}: #{data}" }
        raise NeuralNetTrainError.new(message)
      end
    end

    def validate_values(array : Array(Float64), location : String)
      # Detect exploading gradiants in output
      array.each do |ar|
        if ar.infinite?
          @log.info { "Found an '#{ar}' value, run stopped." }
          puts "#{location}: #{array}"
          puts "Output neurons:"
          puts @output_layers.last.neurons
          raise NeuralNetRunError.new("Exploding gradients detected")
        end
      end

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
      @log.info { "Predicted #{correct} out of #{correct + incorrect} (#{(correct.to_f/(correct + incorrect).to_f)*100}% accuracy)" }
      correct.to_f/(correct + incorrect).to_f
    end
  end
end
