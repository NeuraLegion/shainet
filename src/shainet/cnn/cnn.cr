require "logger"
require "./**"

module SHAInet
  alias CNNLayer = InputLayer | ReluLayer | MaxPoolLayer | FullyConnectedLayer | DropoutLayer | SoftmaxLayer

  # Note: Data is stored within specific classes.
  #       Structure hierarchy: CNN > Layer > Filter > Channel > Row > Neuron/Synapse

  class CNN
    # General network parameters
    getter layers : Array(CNNLayer | ConvLayer) # , :all_neurons, :all_synapses
    getter error_signal : Array(Float64), total_error : Float64, mean_error : Float64, w_gradient : Array(Float64), b_gradient : Array(Float64)

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
      @w_gradient = Array(Float64).new   # Needed for batch train
      @b_gradient = Array(Float64).new   # Needed for batch train

      @learning_rate = 0.7 # Standard parameter for GD
      @momentum = 0.3      # Improved GD

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
                 activation_function : Proc(GenNum, Array(Float64)) = SHAInet.none)
      @layers << ConvLayer.new(@layers.last, filters_num, window_size, stride, padding)
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

    def add_fconnect(l_size : Int32, activation_function : Proc(GenNum, Array(Float64)) = SHAInet.none)
      @layers << FullyConnectedLayer.new(@layers.last, l_size, activation_function)
    end

    def add_softmax
      raise CNNInitializationError.new("Softmax layer must come only after a fully connected layer.") unless @layers.last.class == SHAInet::FullyConnectedLayer
      @layers << SoftmaxLayer.new(@layers.last.as(FullyConnectedLayer))
    end

    def run(input_data : Array(Array(Array(GenNum))), stealth : Bool = true)
      if stealth == false
        puts "############################"
        puts "Starting run..."
      end
      # Activate all layers one by one
      @layers.each do |layer|
        if layer.is_a?(InputLayer)
          layer.as(InputLayer).activate(input_data) # activation of input layer
        else
          layer.activate # activate the rest of the layers
        end
      end
      # Get the result from the output layer
      if stealth == false
        puts "....."
        puts "Network output is: #{@layers.last.as(FullyConnectedLayer | SoftmaxLayer).output}"
        puts "############################"
      end
      return @layers.last.as(FullyConnectedLayer | SoftmaxLayer).output
    end

    def evaluate(input_data : Array(Array(Array(GenNum))), expected_output : Array(GenNum), cost_function : Symbol | String)
      # raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d).") if COST_FUNCTIONS.any? { |x| x == cost_function.to_s } == false

      actual = run(input_data, stealth = true)
      raise NeuralNetRunError.new("Expected and network outputs have different sizes.") unless actual.size == expected_output.size

      # Get the error signal for the final layer, based on the cost function (error gradient is stored in the output neurons)
      @error_signal = [] of Float64 # Set error signal to 0 for current run
      case cost_function.to_s
      when "mse"
        expected_output.size.times do |i|
          neuron = @layers.last.as(FullyConnectedLayer | SoftmaxLayer).filters.first.neurons.first[i] # Update error of all neurons in the output layer based on the actual result
          neuron.gradient = SHAInet.quadratic_cost_derivative(expected_output[i].to_f64, actual[i].to_f64)*neuron.sigma_prime
          @error_signal << SHAInet.quadratic_cost(expected_output[i].to_f64, actual[i].to_f64) # Store the output error based on cost function
          # .as(FullyConnectedLayer | MaxPoolLayer)
        end
      when "c_ent"
        expected_output.size.times do |i|
          neuron = @layers.last.as(FullyConnectedLayer | SoftmaxLayer).filters.first.neurons.first[i]
          neuron.gradient = SHAInet.cross_entropy_cost_derivative(expected_output[i].to_f64, actual[i].to_f64)*neuron.sigma_prime
          # TODO: add support for multiple output layers
          @error_signal << SHAInet.cross_entropy_cost(expected_output[i].to_f64, actual[i].to_f64)
        end
      when "exp"
        # TODO
      when "hel_d"
        # TODO
      when "kld"
        # TODO
      when "gkld"
        # TODO
      when "ita_sai_d"
        # TODO
      end

      @total_error = @error_signal.reduce(0.0) { |acc, i| acc + i } # Sum up all the errors from output layer


    rescue e : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{e}")
    end

    # Online train, updates weights/biases after each data point (stochastic gradient descent)
    def train(data : Array(Array(Array(Array(Array(Float64))) | Array(Float64))), # Input structure: data = [[Input = Array(Array(Array(GenNum)))],[Expected result = Array(GenNum)]]
              training_type : Symbol | String,                                    # Type of training: :sgdm, :rprop, :adam
              cost_function : Symbol | String,                                    # one of COST_FUNCTIONS described at the top of the file
              epochs : Int32,                                                     # a criteria of when to stop the training
              error_threshold : Float64,                                          # a criteria of when to stop the training
              log_each : Int32 = 1000)                                            # determines what is the step for error printout

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

        # Go over each data point and update the weights/biases based on the specific example
        data.each do |data_point|
          # Update error signal, error gradient and total error at the output layer based on current input
          evaluate(data_point[0].as(Array(Array(Array(Float64)))), data_point[1].as(Array(Float64)), cost_function)

          # Propogate the errors backwards through the hidden layers
          @layers.reverse_each do |layer|
            layer.error_prop # Each layer has a different backpropogation function
          end

          # Calculate MSE
          if @error_signal.size == 1
            error_avg = 0.0
          else
            error_avg = @total_error/@layers.last.as(FullyConnectedLayer | SoftmaxLayer).output.size
          end
          sqrd_dists = [] of Float64
          @error_signal.each { |e| sqrd_dists << (e - error_avg)**2 }
          sqr_sum = sqrd_dists.reduce { |acc, i| acc + i }
          @mean_error = sqr_sum/@layers.last.as(FullyConnectedLayer | SoftmaxLayer).output.size

          # Update all wieghts & biases
          # update_weights(training_type, batch = false)
          # update_biases(training_type, batch = false)

          @prev_mean_error = @mean_error
        end
      end
    rescue e : Exception
      @logger.error("Error in training: #{e} #{e.inspect_with_backtrace}")
      raise e
    end

    def update_weights
    end

    # Update weights based on the learning type chosen and layer type
    def update_weights(learn_type : Symbol | String, batch : Bool = false)
      @layers.each do |layer|
      end
      @all_synapses.each_with_index do |synapse, i|
        # Get current gradient
        if batch == true
          synapse.gradient = @w_gradient.not_nil![i]
        else
          synapse.gradient = (synapse.source_neuron.activation)*(synapse.dest_neuron.gradient)
        end

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

            synapse.weight -= synapse.prev_delta_w if @mean_error >= @prev_mean_error

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

          m_hat = synapse.m_current/(1 - (@beta1)**@time_step)
          v_hat = synapse.v_current/(1 - (@beta2)**@time_step)
          synapse.weight -= (@alpha*m_hat)/(v_hat**0.5 + @epsilon)

          synapse.m_prev = synapse.m_current
          synapse.v_prev = synapse.v_current
        end
      end
    end

    # Update biases based on the learning type chosen
    def update_biases(learn_type : Symbol | String, batch : Bool = false)
      @all_neurons.each_with_index do |neuron, i|
        if batch == true
          neuron.gradient = @b_gradient.not_nil![i]
        end

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

            neuron.bias -= neuron.prev_delta_b if @mean_error >= @prev_mean_error

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

          m_hat = neuron.m_current/(1 - (@beta1)**@time_step)
          v_hat = neuron.v_current/(1 - (@beta2)**@time_step)
          neuron.bias -= (@alpha*m_hat)/(v_hat**0.5 + @epsilon)

          neuron.m_prev = neuron.m_current
          neuron.v_prev = neuron.v_current
        end
      end
    end

    def log_summary(e)
      @logger.info("Epoch: #{e}, Total error: #{@total_error}, MSE: #{@mean_error}")
    end
  end

  class DummyLayer
    # This is a place-holder layer, allows explicit network directionality

    @filters : Nil

    def initialize
      @filters = nil
    end
  end
end
