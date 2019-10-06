require "logger"
require "./**"

module SHAInet
  alias CNNLayer = InputLayer | ReluLayer | MaxPoolLayer | FullyConnectedLayer | DropoutLayer | SoftmaxLayer
  alias CNNPair = {input: Array(Array(Array(Float64))), output: Array(Float64)}

  # Note: Data is stored within specific classes.
  # Structure hierarchy: CNN > Layer > Filter > Channel > Row > Neuron/Synapse

  class CNN
    COST_FUNCTIONS = ["mse", "c_ent", "exp", "hel_d", "kld", "gkld", "ita_sai_d"]

    # General network parameters
    getter layers : Array(CNNLayer | ConvLayer)                                       # , :all_neurons, :all_synapses
    getter error_signal : Array(Float64), total_error : Float64, mean_error : Float64 # , w_gradient : Array(Float64), b_gradient : Array(Float64)

    # Parameters for SGD + Momentum
    property learning_rate : Float64, momentum : Float64

    # Parameters for Rprop
    property etah_plus : Float64, etah_minus : Float64, delta_max : Float64, delta_min : Float64
    getter prev_mean_error : Float64

    # Parameters for Adam
    property alpha : Float64
    getter beta1 : Float64, beta2 : Float64, epsilon : Float64, time_step : Int32

    def initialize(@logger : Logger = Logger.new(STDOUT))
      @layers = Array(CNNLayer | ConvLayer).new
      # @all_neurons = Array(Neuron).new
      # @all_synapses = Array(Synapse | CnnSynapse).new

      @error_signal = Array(Float64).new # Array of errors for each neuron in the output layer, based on specific input
      @total_error = Float64.new(1)      # Sum of errors from output layer, based on a specific input
      @mean_error = Float64.new(1)       # MSE of netwrok, based on all errors of output layer for a specific input or batch

      @learning_rate = 0.005 # Standard parameter for GD
      @momentum = 0.05       # Improved GD

      @etah_plus = 1.2                           # For iRprop+ , how to increase step size
      @etah_minus = 0.5                          # For iRprop+ , how to decrease step size
      @delta_max = 50.0                          # For iRprop+ , max step size
      @delta_min = 0.1                           # For iRprop+ , min step size
      @prev_mean_error = rand(0.001..1.0).to_f64 # For iRprop+ , needed for backtracking

      @alpha = 0.001        # For Adam , step size (recomeneded: only change this hyper parameter when fine-tuning)
      @beta1 = 0.9          # For Adam , exponential decay rate (not recommended to change value)
      @beta2 = 0.999        # For Adam , exponential decay rate (not recommended to change value)
      @epsilon = 10**(-8.0) # For Adam , prevents exploding gradients (not recommended to change value)
      @time_step = 0        # For Adam
    end

    def add_input(input_volume : Array(Int32))
      @layers << InputLayer.new(input_volume)
    end

    def add_conv(filters_num : Int32,
                 window_size : Int32,
                 stride : Int32,
                 padding : Int32,
                 activation_function : ActivationFunction = SHAInet.none)
      @layers << ConvLayer.new(self, @layers.last, filters_num, window_size, stride, padding, activation_function)
    end

    def add_relu(l_relu_slope : Float64 = 0.0)
      @layers << ReluLayer.new(@layers.last, l_relu_slope)
    end

    def add_maxpool(pool : Int32, stride : Int32)
      @layers << MaxPoolLayer.new(@layers.last, pool, stride)
    end

    def add_dropout(drop_percent : Int32 = 5)
      @layers << DropoutLayer.new(@layers.last, drop_percent)
    end

    def add_fconnect(l_size : Int32, activation_function : ActivationFunction = SHAInet.none)
      @layers << FullyConnectedLayer.new(master_network: self, prev_layer: @layers.last, l_size: l_size, activation_function: activation_function)
    end

    def add_softmax(range : Range(Int32, Int32) = (0..-1))
      @layers << SoftmaxLayer.new(@layers.last.as(FullyConnectedLayer | ReluLayer), range: range)
    end

    def run(input_data : Array(Array(Array(GenNum))), stealth : Bool = true) : Array(Float64)
      if stealth == false
        puts "############################"
        puts "Starting run..."
      end
      # Activate all layers one by onelayer.inspect("activations")
      @layers.each do |layer|
        if layer.is_a?(InputLayer)
          layer.as(InputLayer).activate(input_data) # activation of input layer
        else
          layer.activate # activate the rest of the layers

          if stealth == false
            layer.inspect("activations")
          end
        end
      end
      # Get the result from the output layer
      if stealth == false
        puts "....."
        puts "Network output is: #{@layers.last.as(FullyConnectedLayer | SoftmaxLayer).output}"
        puts "############################"
      end
      @layers.last.as(FullyConnectedLayer | SoftmaxLayer).output
    end

    def evaluate(input_data : Array(Array(Array(GenNum))),
                 expected_output : Array(GenNum),
                 cost_function : CostFunction = SHAInet.quadratic_cost)
      #
      actual_output = run(input_data, stealth: true)

      raise NeuralNetRunError.new(
        "Expected and network outputs have different sizes.") unless actual_output.size == expected_output.size
      # Stop scan if we have NaNs in the output
      actual_output.each { |ar| raise NeuralNetRunError.new(
        "Found a NaN value, run stopped.\noutput:#{actual_output}") if ar.nan? }

      # Get the error signal for the final layer, based on the cost function (error gradient is stored in the output neurons)
      @error_signal = [] of Float64 # collect all the errors for current run
      actual_output.size.times do |i|
        neuron = @layers.last.as(FullyConnectedLayer | SoftmaxLayer).filters.first.neurons.first[i] # Output neuron
        cost = cost_function.call(expected_output[i], actual_output[i])
        neuron.gradient = cost[:derivative]*neuron.sigma_prime # Update error of all neurons in the output layer based on the actual result
        @error_signal << cost[:value]                          # Store the output error based on cost function
      end
      @total_error = @error_signal.reduce(0.0) { |acc, i| acc + i } # Sum up all the errors from output layer


    rescue e : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{e}")
    end

    # Online train, updates weights/biases after each data point (stochastic gradient descent)
    def train(data : NamedTuple(input: Array(Array(Array(Float64))), output: Array(Float64)), # Input structure: data = [[Input = Array(Array(Array(GenNum)))],[Expected result = Array(GenNum)]]
              training_type : Symbol | String,                                                # Type of training: :sgdm, :rprop, :adam
              cost_function : Symbol | String | CostFunction = :mse,                          # one of COST_FUNCTIONS described at the top of the file or Proc
              epochs : Int32 = 1,                                                             # a criteria of when to stop the training
              error_threshold : Float64 = 0.0,                                                # a criteria of when to stop the training
              log_each : Int32 = 1000)                                                        # determines what is the step for error printout

      # verify_data(data)
      @logger.info("Training started")
      loop do |e|
        if e % log_each == 0
          log_summary(e)
        end
        if e >= epochs || (error_threshold >= @mean_error) && (e > 0)
          log_summary(e)
          break
        end

        # Change String/Symbol into the corrent proc
        if cost_function.is_a?(Symbol) || cost_function.is_a?(String)
          raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d).") if COST_FUNCTIONS.any? { |x| x == cost_function.to_s } == false
          proc = get_cost_proc(cost_function.to_s)
          cost_function = proc
        end

        # Go over each data point and update the weights/biases based on the specific example
        data.each do |data_point|
          # Update error signal, error gradient and total error at the output layer based on current input
          evaluate(data_point[:input], data_point[:output], cost_function)

          # Propogate the errors backwards through the hidden layers
          @layers.reverse_each do |layer|
            layer.error_prop(batch: false) # Each layer has a different backpropogation function
          end

          # Calculate MSE
          if @error_signal.size == 1
            error_avg = 0.0
          else
            error_avg = @total_error/@layers.last.as(FullyConnectedLayer | SoftmaxLayer).output.size
          end
          sqrd_dists = [] of Float64
          @error_signal.each { |error| sqrd_dists << (error - error_avg)**2 }
          sqr_sum = sqrd_dists.reduce { |acc, i| acc + i }
          @mean_error = sqr_sum/@layers.last.as(FullyConnectedLayer | SoftmaxLayer).output.size

          # Update all weights & biases
          update_wb(training_type, batch: false)

          # update_biases(training_type, batch = false)

          @prev_mean_error = @mean_error
        end
      end
    rescue e : Exception
      @logger.error("Error in training: #{e} #{e.inspect_with_backtrace}")
      raise e
    end

    # Batch train, updates weights/biases using a gradient sum from all data points in the batch (using gradient descent)
    # ameba:disable Metrics/CyclomaticComplexity
    def train_batch(data : Array(CNNPair), # Input structure: data = [[Input = Array(Array(Array(GenNum)))],[Expected result = Array(GenNum)]]
                    # data : Array(NamedTuple(input: Array(Array(Array(Float64))), output: Array(Float64))), # Input structure: data = [[Input = Array(Array(Array(GenNum)))],[Expected result = Array(GenNum)]]
                    training_type : Symbol | String,                       # Type of training: :sgdm, :rprop, :adam
                    cost_function : Symbol | String | CostFunction = :mse, # one of COST_FUNCTIONS described at the top of the file or Proc
                    epochs : Int32 = 1,                                    # a criteria of when to stop the training
                    error_threshold : Float64 = 0.0,                       # a criteria of when to stop the training
                    log_each : Int32 = 1000,                               # determines what is the step for error printout
                    mini_batch_size : Int32 | Nil = nil)
      #
      time_start = Time.new
      @logger.info("Training started")
      batch_size = mini_batch_size ? mini_batch_size : data.size
      @time_step = 0

      # Change String/Symbol into the corrent proc
      if cost_function.is_a?(Symbol) || cost_function.is_a?(String)
        raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d).") if COST_FUNCTIONS.any? { |x| x == cost_function.to_s } == false
        proc = get_cost_proc(cost_function.to_s)
        cost_function = proc
      end

      loop do |epoch|
        slice_num = 1
        if epoch % log_each == 0
          log_summary(epoch)
          # @all_neurons.each { |s| puts s.gradient }
        end
        if epoch >= epochs || (error_threshold >= @mean_error) && (epoch > 1)
          log_summary(epoch)
          break
        end

        data.each_slice(batch_size, reuse: false) do |data_slice|
          verify_data(data_slice)
          time_now = Time.new
          @logger.info("Mini-batch # #{slice_num}| Mini-batch size: #{batch_size} | Runtime: #{time_now - time_start}") if mini_batch_size
          slice_num += 1
          @time_step += 1 if mini_batch_size # in mini-batch update adam time_step

          batch_mean = [] of Float64
          all_errors = [] of Float64

          # Go over each data point and collect gradients of weights/biases based on each specific example
          data_slice.each do |data_point|
            # Update error signal, error gradient and total error at the output layer based on current input

            evaluate(data_point[:input], data_point[:output], cost_function)
            all_errors << @total_error

            # Propogate the errors backwards through the hidden layers
            @layers.reverse_each do |layer|
              layer.error_prop(batch: true)
            end

            # Calculate MSE per data point
            if @error_signal.size == 1
              error_avg = 0.0
            else
              error_avg = @total_error/@layers.last.as(FullyConnectedLayer | SoftmaxLayer).output.size
            end
            sqrd_dists = [] of Float64
            @error_signal.each { |e| sqrd_dists << (e - error_avg)**2 }

            @mean_error = (sqrd_dists.reduce { |acc, i| acc + i })/@layers.last.as(FullyConnectedLayer | SoftmaxLayer).output.size
            batch_mean << @mean_error
          end

          # Total error per batch
          @total_error = all_errors.reduce { |acc, i| acc + i }

          # Calculate MSE per batch
          @mean_error = (batch_mean.reduce { |acc, i| acc + i })/data_slice.size

          # Update all wieghts & biases for the batch
          @time_step += 1 unless mini_batch_size # Based on how many epochs have passed in current training run, needed for Adam

          # Update all wieghts & biases
          update_wb(training_type, batch: true)

          @prev_mean_error = @mean_error
        end
      end
    end

    # Go over all layers and update the weights and biases, based on learning type chosen
    def update_wb(training_type : Symbol | String, batch : Bool = false)
      @layers.each do |layer|
        layer.update_wb(training_type, batch) # Each layer does this function differently
      end
    end

    def check_nan : Bool
      @layers.last.as(FullyConnectedLayer | SoftmaxLayer).output.each do |value|
        return value.nan?
      end
    end

    def output : Array(Float64)
      @layers.last.as(FullyConnectedLayer | SoftmaxLayer).output
    end

    def propagate_backwards
      @layers.reverse_each do |layer|
        layer.error_prop(batch: false)
      end
    end

    # Update the output layer gradients manually
    def update_output_gradients(cost_function_derivatives : Array(Float64))
      output_neurons = @layers.last.as(FullyConnectedLayer | SoftmaxLayer).all_neurons

      unless new_gradients.size == output_neurons.size
        raise NeuralNetRunError.new("New gradients array must be the same size as the output layer.")
      end

      output_neurons.each_with_index do |neuron, i|
        neuron.gradient = cost_function_derivatives[i]*neuron.sigma_prime
      end
    end

    def verify_data(data : Array(CNNPair))
      message = nil

      random_input = data.sample[:input].size
      random_output = data.sample[:output].size
      data.each_with_index do |test, i|
        if (test[:input].size != random_input)
          message = "Input data sizes are inconsistent"
        end
        if (test[:output].size != random_output)
          message = "Output data sizes are inconsistent"
        end
        unless (test[:output].size == @layers.last.as(FullyConnectedLayer | SoftmaxLayer).filters[0].neurons[0].size)
          message = "data at index #{i} and size: #{test[:output].size} mismatch output layer size"
        end
      end
      if message
        @logger.error("#{message}: #{data}")
        raise NeuralNetTrainError.new(message)
      end
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

    def inspect(what : String | Symbol)
      case what.to_s
      when "weights"
        @layers.each { |layer| layer.inspect("weights") }
      when "bias"
        @layers.each { |layer| layer.inspect("bias") }
      when "activations"
        @layers.each { |layer| layer.inspect("activations") }
      when "gradients"
        @layers.each { |layer| layer.inspect("gradients") }
        puts "------------------------------------------------"
      end
    end

    def log_summary(e)
      @logger.info("Epoch: #{e}, Total error: #{@total_error}, MSE: #{@mean_error}")
    end
  end
end
