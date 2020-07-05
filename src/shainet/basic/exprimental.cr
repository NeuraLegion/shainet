require "log"

module SHAInet
  # ------------
  # This is an experimental file for pointer matrix implementation
  # ------------
  class Layer
    def propagate_forward_exp(prev_layer : Layer)
      # puts "@weights:"
      # @weights.show
      # puts "prev_layer.activations:"
      # prev_layer.activations.show
      # puts "@input_sums:"
      # @input_sums.show

      @weights.static_dot(prev_layer.activations, @input_sums)

      @input_sums + @biases
      @input_sums.data[0].each_with_index do |input_sum, i|
        # @input_sums[i].value = input_sum
        @activations.data[0][i].value, @sigma_primes.data[0][i].value = activation_function.call(input_sum.value)
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error in propagate_forward_exp: #{e} ")
    end
  end

  class Network
    def run_exp(input : Array(GenNum), stealth : Bool = false) : Array(Float64)
      verify_net_before_train
      raise NeuralNetRunError.new(
        "Error input data size: #{input.size} doesn't fit input layer size: #{@input_layers.first.neurons.size}.") unless input.size == @input_layers.first.neurons.size

      # Insert the input data into the input layer
      input.each_with_index do |data, i|
        # Inserts the input information into the input layers
        # TODO: add support for multiple input layers
        @input_layers.last.activations.data[i][0].value = data.to_f64
      end

      unless stealth # Hide report during training
        @input_layers.last.activations.show("input layer activations:")
      end

      # Propogate the information forward through the hidden layers
      @hidden_layers.size.times do |l|
        if l == 0
          @hidden_layers[l].propagate_forward_exp(@input_layers.last)
        else
          @hidden_layers[l].propagate_forward_exp(@hidden_layers[l - 1])
        end

        unless stealth # Hide report during training
          @hidden_layers[l].activations.show("hidden layer #{l} activations:")
          @hidden_layers[l].weights.show("hidden layer #{l} weights:")
        end
      end

      # Propogate the information through the output layers
      @output_layers.each { |l| l.propagate_forward_exp(@hidden_layers.last) }

      @output_layers.last.neurons.map { |neuron| neuron.activation } # return an array of all output neuron activations
      # TODO: add support for multiple output layers


    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    def evaluate_exp(input_data : Array(GenNum),
                     expected_output : Array(GenNum),
                     cost_function : CostFunction = SHAInet.quadratic_cost,
                     stealth : Bool = true)
      #
      # puts "input_data: #{input_data}"
      # puts "expected_output: #{expected_output}"
      actual_output = run_exp(input_data, stealth: stealth)
      # puts "actual_output: #{actual_output}"

      # unless stealth # Hide output report during training
      # msg = "Input & Output vs. Expected:"
      # msg += "\n  Input:\n  #{input_data}"
      # msg += "\n  Output:\n  #{actual_output}"
      # msg += "\n  Expected:\n  #{expected_output}"
      # Log.info(msg)
      # end

      # Detect exploading gradiants and NaNs in output
      validate_values(actual_output, "actual_output")
      # actual_output.each do |ar|
      #   if ar.infinite?
      #     Log.info("Found an '#{ar}' value, run stopped.")
      #     puts "output:#{actual_output}"
      #     puts "Output neurons:"
      #     puts @output_layers.last.neurons
      #     raise NeuralNetRunError.new("Exploding gradients detected")
      #   end
      # end

      # # Detect NaNs in output
      # actual_output.each { |ar| raise NeuralNetRunError.new(
      #   "Found a NaN value, run stopped.\noutput:#{actual_output}") if ar.nan? }

      # Get the error signal for the final layer, based on the cost function (error gradient is stored in the output neurons)
      @error_signal = [] of Float64 # Collect all the errors for current run

      actual_output.size.times do |i|
        neuron = @output_layers.last.neurons[i] # Update error of all neurons in the output layer based on the actual result
        cost = cost_function.call(expected_output[i], actual_output[i])
        neuron.gradient = cost[:derivative]*neuron.sigma_prime
        @error_signal << cost[:value] # Store the output error based on cost function

        # puts "Actual output: #{actual_output}"
        # puts "Cost value: #{cost[:value]}"
        # puts "cost derivative: #{cost[:derivative]}"
        # puts "---"
      end
      # @error_signal = Array(Float64).new(actual_output.size) { 0.0 }

      # puts "---- @error_signal ----"
      # puts @error_signal
      # puts "-----------------------"
      validate_values(@error_signal, "error_signal")

      @total_error = @error_signal.reduce(0.0) { |acc, i| acc + i } # Sum up all the errors from output layer

      # puts "@error_signal: #{@error_signal}"
      # puts "@total_error: #{@total_error}"


    rescue e : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{e}")
    end

    def train_es_exp(data : Array(Array(Array(GenNum))) | SHAInet::TrainingData,   # Input structure: data = [[Input = [] of Float64],[Expected result = [] of Float64]]
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
        raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d).") if COST_FUNCTIONS.any? { |x| x == cost_function.to_s } == false
        proc = get_cost_proc(cost_function.to_s)
        cost_function = proc
      end

      counter = 0_i64
      loop do
        # Autosave the network
        unless autosave.nil?
          if counter % autosave[:freq] == 0 && (counter > 0)
            # Log.info("Network saved.")
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
        i = 0
        slices = (data.size.to_f64 / mini_batch_size).ceil.to_i

        raw_data.each_slice(batch_size, reuse = false) do |data_slice|
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
              evaluate_exp(data_point[0], data_point[1], cost_function)
              update_mse
              batch_mse_sum += @mse
              @error_signal.size.times { |i| batch_errors_sum[i] += @error_signal[i] }
            end

            @mse = (batch_mse_sum / mini_batch_size) # Update MSE of the batch
            @error_signal = batch_errors_sum
            organism.update_reward
          end

          pool.pull_params
          # puts "pulled params"
          # puts "Output neurons:"
          # @output_layers.last.neurons.each { |n| puts "n.activation: #{n.activation}, n.gradient #{n.gradient}" }

          # Show training progress of the mini-batches
          i += 1
          if counter % log_each == 0
            Log.info { "  Slice: (#{i} / #{slices}), MSE: #{@mse}" } if show_slice
            # Log.info("@error_signal: #{@error_signal}")
          end
        end
        counter += 1
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error in run_es: #{e.inspect_with_backtrace}")
    end
  end
end
