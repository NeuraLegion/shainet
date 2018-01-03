require "logger"
require "json"

module SHAInet
  class Network
    # # Notes:
    # # ------------
    # # There are no matrices in this implementation, instead the gradient values are stored in each neuron/synapse independently.
    # # When preforming propogation, all the math is done iteratively on each neuron/synapse locally.

    LAYER_TYPES      = [:input, :hidden, :output]
    CONNECTION_TYPES = [:full, :ind_to_ind, :random]
    COST_FUNCTIONS   = [:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d]

    # General network parameters
    getter :input_layers, :output_layers, :hidden_layers, :all_neurons, :all_synapses
    getter error_signal : Array(Float64), total_error : Float64, :mean_error, w_gradient : Array(Float64), b_gradient : Array(Float64)

    # Parameters for SGD + Momentum
    property learning_rate : Float64, momentum : Float64

    # Parameters for Rprop
    property etah_plus : Float64, etah_minus : Float64, delta_max : Float64, delta_min : Float64
    getter prev_total_error : Float64

    # Parameters for Adam
    property alpha : Float64
    getter beta1 : Float64, beta2 : Float64, epsilon : Float64, time_step : Int32

    # First creates an empty shell of the entire network
    def initialize(@logger : Logger = Logger.new(STDOUT))
      @input_layers = Array(Layer).new
      @output_layers = Array(Layer).new
      @hidden_layers = Array(Layer).new
      @all_neurons = Array(Neuron).new   # Array of all current neurons in the network
      @all_synapses = Array(Synapse).new # Array of all current synapses in the network
      @error_signal = Array(Float64).new # Array of errors for each neuron in the output layers, based on specific input
      @total_error = Float64.new(1)      # Sum of errors from output layer, based on a specific input
      @mean_error = Float64.new(1)       # MSE of netwrok, based on all errors of output layer fort a specific input or batch
      @w_gradient = Array(Float64).new   # Needed for batch train
      @b_gradient = Array(Float64).new   # Needed for batch train

      @learning_rate = 0.7 # Standard parameter for GD
      @momentum = 0.3      # Improved GD

      @etah_plus = 1.2                          # For iRprop+ , how to increase step size
      @etah_minus = 0.5                         # For iRprop+ , how to decrease step size
      @delta_max = 50.0                         # For iRprop+ , max step size
      @delta_min = 0.1                          # For iRprop+ , min step size
      @prev_total_error = rand(0.0..1.0).to_f64 # For iRprop+ , needed for backtracking

      @alpha = Float64.new(0.001)        # For Adam , step size (recomeneded: only change this hyper parameter when fine-tuning)
      @beta1 = Float64.new(0.9)          # For Adam , exponential decay rate (not recommended to change value)
      @beta2 = Float64.new(0.999)        # For Adam , exponential decay rate (not recommended to change value)
      @epsilon = Float64.new(10**(-8.0)) # For Adam , prevents exploding gradients (not recommended to change value)
      @time_step = 0                     # For Adam
    end

    # Create and populate a layer with neurons
    # l_type is: :input, :hidden or :output
    # l_size = how many neurons in the layer
    # n_type = advanced option for different neuron types
    def add_layer(l_type : Symbol, l_size : Int32, n_type : String = "memory", activation_function : Proc(GenNum, Array(Float64)) = SHAInet.sigmoid)
      layer = Layer.new(n_type, l_size, activation_function, @logger)
      layer.neurons.each { |neuron| @all_neurons << neuron } # To easily access neurons later

      case l_type
      when :input
        @input_layers << layer
      when :hidden
        @hidden_layers << layer
      when :output
        if @output_layers.empty?
          @output_layers << layer
        else
          @hidden_layers << @output_layers.first
          @output_layers.delete(@output_layers.first)
          @output_layers << layer
          connect_ltl(@hidden_layers.last, @output_layers.first, :full)
        end
      else
        raise NeuralNetRunError.new("Must define correct layer type (:input, :hidden, :output).")
      end
    end

    def clean_dead_neurons
      current_neuron_number = @all_neurons.size
      @hidden_layers.each do |h_l|
        h_l.neurons.each do |neuron|
          kill = false
          if neuron.bias == 0
            neuron.synapses_in.each do |s|
              if s.weight == 0
                kill = true
              end
            end
          end
          if kill
            # Kill neuron and all connected synapses
            neuron.synapses_in.each { |s| @all_synapses.delete(s) }
            neuron.synapses_out.each { |s| @all_synapses.delete(s) }
            @all_neurons.delete(neuron)
            h_l.neurons.delete(neuron)
          end
        end
      end
      @logger.info("Cleaned #{current_neuron_number - @all_neurons.size} dead neurons")
    end

    def verify_net_before_train
      if @input_layers.empty?
        raise NeuralNetRunError.new("No input layers defined")
      elsif @hidden_layers.empty?
        raise NeuralNetRunError.new("Need atleast one hidden layer")
      elsif @output_layers.empty?
        raise NeuralNetRunError.new("No output layers defined")
      end
    end

    # Connect all the layers in order (input and output don't connect between themselves): input, hidden, output
    def fully_connect
      # Connect all input layers to the first hidden layer
      @input_layers.each do |source|
        connect_ltl(source, @hidden_layers.first, :full)
      end

      # Connect all hidden layer between each other hierarchically
      @hidden_layers.size.times do |index|
        next if index + 2 > @hidden_layers.size
        connect_ltl(@hidden_layers[index], @hidden_layers[index + 1], :full)
      end

      # Connect last hidden layer to all output layers
      @output_layers.each do |layer|
        connect_ltl(@hidden_layers.last, layer, :full)
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error fully connecting network: #{e}")
    end

    # Connect two specific layers with synapses
    def connect_ltl(source : Layer, destination : Layer, connection_type : Symbol)
      raise NeuralNetInitalizationError.new("Error initilizing network, must choose correct connection type.") if CONNECTION_TYPES.any? { |x| x == connection_type } == false
      case connection_type
      # Connect each neuron from source layer to all neurons in destination layer
      when :full
        source.neurons.each do |neuron1|        # Source neuron
          destination.neurons.each do |neuron2| # Destination neuron
            synapse = Synapse.new(neuron1, neuron2)
            neuron1.synapses_out << synapse
            neuron2.synapses_in << synapse
            @all_synapses << synapse
          end
        end
        # Connect each neuron from source layer to neuron with corresponding index in destination layer
      when :ind_to_ind
        raise NeuralNetInitalizationError.new("Error initializing network, index to index connection requires layers of same size.") if source.neurons.size != destination.neurons.size
        (0..source.neurons.size).each do |index|
          synapse = Synapse.new(source.neurons[index], destination.neurons[index])
          source.neurons[index].synapses_out << synapse
          destination.neurons[index].synapses_in << synapse
          @all_synapses << synapse
        end

        # Randomly decide if each neuron from source layer will connect to a neuron from destination layer
      when :random
        source.neurons.each do |neuron1|        # Source neuron
          destination.neurons.each do |neuron2| # Destination neuron
            x = rand(0..1)
            if x <= 0.5 # Currently set to 50% chance, this can be changed at will
              synapse = Synapse.new(neuron1, neuron2)
              neuron1.synapses_out << synapse
              neuron2.synapses_in << synapse
              @all_synapses << synapse
            end
          end
        end
      end
      @all_synapses.uniq!
    end

    # Run an input throught the network to get an output (weights & biases do not change)
    def run(input : Array(GenNum), stealth : Bool = false) : Array(Float64)
      verify_net_before_train
      raise NeuralNetRunError.new("Error input data size: #{input.size} doesn't fit input layer size: #{@input_layers.first.neurons.size}.") unless input.size == @input_layers.first.neurons.size

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
        @logger.info("Input => #{input}, network output => #{output}")
      end
      output
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.inspect_with_backtrace}")
    end

    # Quantifies how good the network performed for a single input compared to the expected output
    # This function returns the actual output and updates the error gradient for the output layer
    def evaluate(input : Array(GenNum), expected : Array(GenNum), cost_function : Symbol)
      raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d).") if COST_FUNCTIONS.any? { |x| x == cost_function } == false

      actual = run(input, stealth = true)
      # Get the error signal for the final layer, based on the cost function (error gradient is stored in the output neurons)
      @error_signal = [] of Float64
      case cost_function
      when :mse
        expected.size.times do |i|
          neuron = @output_layers.last.neurons[i] # Update error of all neurons in the output layer based on the actual result
          neuron.gradient = SHAInet.quadratic_cost_derivative(expected[i].to_f64, actual[i].to_f64)*neuron.sigma_prime
          # TODO: add support for multiple output layers
          @error_signal << SHAInet.quadratic_cost(expected[i].to_f64, actual[i].to_f64) # Store the output error based on cost function
        end
      when :c_ent
        expected.size.times do |i|
          neuron = @output_layers.last.neurons[i]
          neuron.gradient = SHAInet.cross_entropy_cost_derivative(expected[i].to_f64, actual[i].to_f64)*neuron.sigma_prime
          # TODO: add support for multiple output layers
          @error_signal << SHAInet.cross_entropy_cost(expected[i].to_f64, actual[i].to_f64)
        end
      when :exp
        # TODO
      when :hel_d
        # TODO
      when :kld
        # TODO
      when :gkld
        # TODO
      when :ita_sai_d
        # TODO
      end

      @total_error = @error_signal.reduce(0.0) { |acc, i| acc + i } # Sum up all the errors from output layer


    rescue e : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{e}")
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
        @logger.error("#{message}: #{data}")
        raise NeuralNetTrainError.new(message)
      end
    end

    def log_summery(e)
      @logger.info("Epoch: #{e}, Total error: #{@total_error}, MSE: #{@mean_error}")
    end

    # Online train, updates weights/biases after each data point (stochastic gradient descent)
    def train(data : Array(Array(Array(GenNum))), # Input structure: data = [[Input = [] of Float64],[Expected result = [] of Float64]]
              training_type : Symbol,             # Type of training: :sgdm, :rprop, :adam
              cost_function : Symbol,             # one of COST_FUNCTIONS described at the top of the file
              epochs : Int32,                     # a criteria of when to stop the training
              error_threshold : Float64,          # a criteria of when to stop the training
              log_each : Int32 = 1000)            # determines what is the step for error printout

      verify_data(data)
      @logger.info("Training started")
      loop do |e|
        if e % log_each == 0
          log_summery(e)
        end
        if e >= epochs || (error_threshold >= @mean_error) && (e > 0)
          log_summery(e)
          break
        end

        # Go over each data point and update the weights/biases based on the specific example
        data.each do |data_point|
          # Update error signal, error gradient and total error at the output layer based on current input
          evaluate(data_point[0], data_point[1], cost_function)

          # Propogate the errors backwards through the hidden layers
          @hidden_layers.each do |l|
            l.neurons.each { |neuron| neuron.hidden_error_prop } # Update neuron error based on errors*weights of neurons from the next layer
          end

          # Calculate MSE
          if @error_signal.size == 1
            error_avg = 0.0
          else
            error_avg = @total_error/@output_layers.last.neurons.size
          end
          sqrd_dists = [] of Float64
          @error_signal.each { |e| sqrd_dists << (e - error_avg)**2 }
          sqr_sum = sqrd_dists.reduce { |acc, i| acc + i }
          @mean_error = sqr_sum/@output_layers.last.neurons.size

          # Update all wieghts & biases
          update_weights(training_type, batch = false)
          update_biases(training_type, batch = false)

          @prev_total_error = @total_error
        end
      end
    rescue e : Exception
      @logger.error("Error in training: #{e} #{e.inspect_with_backtrace}")
      raise e
    end

    # Batch train, updates weights/biases using a gradient sum from all data points in the batch (using gradient descent)
    def train_batch(data : Array(Array(Array(GenNum))), # Input structure: data = [[Input = [] of Float64],[Expected result = [] of Float64]]
                    training_type : Symbol,             # Type of training: :sgdm, :rprop, :adam
                    cost_function : Symbol,             # one of COST_FUNCTIONS described at the top of the file
                    epochs : Int32,                     # a criteria of when to stop the training
                    error_threshold : Float64,          # a criteria of when to stop the training
                    log_each : Int32 = 1000,            # determines what is the step for error printout
                    mini_batch_size : Int32 | Nil = nil)
      @logger.info("Training started")
      batch_size = mini_batch_size ? mini_batch_size : data.size
      @time_step = 0
      data.each_slice(batch_size, reuse = false) do |data_slice|
        verify_data(data_slice)
        @logger.info("Working on mini-batch size: #{batch_size}") if mini_batch_size
        @time_step += 1 if mini_batch_size # in mini-batch update adam time_step
        loop do |e|
          if e % log_each == 0
            log_summery(e)
          end
          if e >= epochs || (error_threshold >= @mean_error) && (e > 0)
            log_summery(e)
            break
          end
          batch_mean = [] of Float64
          all_errors = [] of Float64
          @w_gradient = Array(Float64).new(@all_synapses.size) { 0.0 } # Save gradients from entire batch before updating weights & biases
          @b_gradient = Array(Float64).new(@all_neurons.size) { 0.0 }

          # Go over each data point and collect gradients of weights/biases based on each specific example
          data_slice.each do |data_point|
            evaluate(data_point[0], data_point[1], cost_function) # Get error gradient from output layer based on current input
            all_errors << @total_error
            # Propogate the errors backwards through the hidden layers
            @hidden_layers.each do |l|
              l.neurons.each { |neuron| neuron.hidden_error_prop } # Update neuron error based on errors*weights of neurons from the next layer
            end

            # Sum all gradients from each data point for the batch update
            @all_synapses.each_with_index { |synapse, i| @w_gradient[i] += (synapse.source_neuron.activation)*(synapse.dest_neuron.gradient) }
            @all_neurons.each_with_index { |neuron, i| @b_gradient[i] += neuron.bias }

            # Calculate MSE per data point
            if @error_signal.size == 1
              error_avg = 0.0
            else
              error_avg = @total_error/@output_layers.last.neurons.size
            end
            sqrd_dists = [] of Float64
            @error_signal.each { |e| sqrd_dists << (e - error_avg)**2 }

            @mean_error = (sqrd_dists.reduce { |acc, i| acc + i })/@output_layers.last.neurons.size
            batch_mean << @mean_error
          end

          @total_error = all_errors.reduce { |acc, i| acc + i }

          # Calculate MSE per batch
          batch_mean = (batch_mean.reduce { |acc, i| acc + i })/data_slice.size
          @mean_error = batch_mean

          # Update all wieghts & biases for the batch
          @time_step += 1 unless mini_batch_size # Based on how many epochs have passed in current training run, needed for Adam
          update_weights(training_type, batch = true)
          update_biases(training_type, batch = true)

          @prev_total_error = @total_error
        end
      end
    end

    # Update weights based on the learning type chosen
    def update_weights(learn_type : Symbol, batch : Bool = false)
      @all_synapses.each_with_index do |synapse, i|
        # Get current gradient
        if batch == true
          synapse.gradient = @w_gradient.not_nil![i]
        else
          synapse.gradient = (synapse.source_neuron.activation)*(synapse.dest_neuron.gradient)
        end

        case learn_type
        # Update weights based on the gradients and delta rule (including momentum)
        when :sgdm
          delta_weight = (-1)*@learning_rate*synapse.gradient + @momentum*(synapse.weight - synapse.prev_weight)
          synapse.weight += delta_weight
          synapse.prev_weight = synapse.weight

          synapse.weight += delta_weight
          synapse.prev_weight = synapse.weight

          # Update weights based on Resilient backpropogation (Rprop), using the improved varient iRprop+
        when :rprop
          if synapse.prev_gradient*synapse.gradient > 0
            delta = [@etah_plus*synapse.prev_delta, @delta_max].min
            delta_weight = (-1)*SHAInet.sign(synapse.gradient)*delta

            synapse.weight += delta_weight
            synapse.prev_weight = synapse.weight
            synapse.prev_delta = delta
            synapse.prev_delta_w = delta_weight
          elsif synapse.prev_gradient*synapse.gradient < 0
            delta = [@etah_minus*synapse.prev_delta, @delta_min].max

            synapse.weight -= synapse.prev_delta_w if @total_error > @prev_total_error

            synapse.prev_gradient = 0.0
            synapse.prev_delta = delta
          elsif synapse.prev_gradient*synapse.gradient == 0
            delta_weight = (-1)*SHAInet.sign(synapse.gradient)*synapse.prev_delta

            synapse.weight += delta_weight
            synapse.prev_delta_w = delta_weight
          end
          # Update weights based on Adaptive moment estimation (Adam)
        when :adam
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
    def update_biases(learn_type : Symbol, batch : Bool = false)
      @all_neurons.each_with_index do |neuron, i|
        if batch == true
          neuron.gradient = @b_gradient.not_nil![i]
        end

        case learn_type
        # Update biases based on the gradients and delta rule (including momentum)
        when :sgdm
          delta_bias = (-1)*@learning_rate*(neuron.gradient) + @momentum*(neuron.bias - neuron.prev_bias)
          neuron.bias += delta_bias
          neuron.prev_bias = neuron.bias

          # Update weights based on Resilient backpropogation (Rprop), using the improved varient iRprop+
        when :rprop
          if neuron.prev_gradient*neuron.gradient > 0
            delta = [@etah_plus*neuron.prev_delta, @delta_max].min
            delta_bias = (-1)*SHAInet.sign(neuron.gradient)*delta

            neuron.bias += delta_bias
            neuron.prev_bias = neuron.bias
            neuron.prev_delta = delta
            neuron.prev_delta_b = delta_bias
          elsif neuron.prev_gradient*neuron.gradient < 0
            delta = [@etah_minus*neuron.prev_delta, @delta_min].max

            neuron.bias -= neuron.prev_delta_b if @total_error > @prev_total_error

            neuron.prev_gradient = 0.0
            neuron.prev_delta = delta
          elsif neuron.prev_gradient*neuron.gradient == 0
            delta_bias = (-1)*SHAInet.sign(neuron.gradient)*neuron.prev_delta

            neuron.bias += delta_bias
            neuron.prev_delta_b = delta_bias
          end
          # Update weights based on Adaptive moment estimation (Adam)
        when :adam
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

    def randomize_all_weights
      raise NeuralNetRunError.new("Cannot randomize weights without synapses") if @all_synapses.empty?
      @all_synapses.each &.randomize_weight
    end

    def randomize_all_biases
      raise NeuralNetRunError.new("Cannot randomize biases without synapses") if @all_synapses.empty?
      @all_neurons.each &.randomize_bias
    end

    def save_to_file(file_path : String)
      dump_network = Array(Hash(String, String | Array(Hash(String, Array(Hash(String, String | Float64)) | Float64 | String | String)))).new

      [@input_layers, @output_layers, @hidden_layers].flatten.each do |layer|
        dump_layer = Hash(String, String | Array(Hash(String, Array(Hash(String, String | Float64)) | Float64 | String | String))).new
        dump_neurons = Array(Hash(String, Array(Hash(String, String | Float64)) | Float64 | String | String)).new
        layer.neurons.each do |neuron|
          n = Hash(String, Array(Hash(String, String | Float64)) | Float64 | String | String).new
          n["id"] = neuron.id
          n["activation"] = neuron.activation
          n["n_type"] = neuron.n_type.to_s
          n["synapses_in"] = Array(Hash(String, String | Float64)).new
          n["synapses_out"] = Array(Hash(String, String | Float64)).new
          neuron.synapses_in.each do |s|
            s_h = Hash(String, String | Float64).new
            s_h["source"] = s.source_neuron.id
            s_h["destination"] = s.dest_neuron.id
            s_h["weight"] = s.weight
            n["synapses_in"].as(Array(Hash(String, String | Float64))) << s_h
          end
          neuron.synapses_out.each do |s|
            s_h = Hash(String, String | Float64).new
            s_h["source"] = s.source_neuron.id
            s_h["destination"] = s.dest_neuron.id
            s_h["weight"] = s.weight
            n["synapses_out"].as(Array(Hash(String, String | Float64))) << s_h
          end
          dump_neurons << n
        end

        l_type = ""
        if @input_layers.includes?(layer)
          l_type = "input"
        elsif @hidden_layers.includes?(layer)
          l_type = "hidden"
        else
          l_type = "output"
        end

        dump_layer["l_type"] = l_type
        dump_layer["neurons"] = dump_neurons
        dump_layer["activation_function"] = layer.activation_function.to_s
        dump_network << dump_layer
      end
      File.write(file_path, {"layers" => dump_network}.to_json)
      @logger.info("Network saved to: #{file_path}")
    end

    def load_from_file(file_path : String)
      net = NetDump.from_json(File.read(file_path))
      net.layers.each do |layer|
        l = Layer.new("memory", 0)
        layer.neurons.each do |neuron|
          n = Neuron.new(neuron.n_type, neuron.id)
          n.activation = neuron.activation
          l.neurons << n
          @all_neurons << n
        end
        case layer.l_type
        when "input"
          @input_layers << l
        when "output"
          @output_layers << l
        when "hidden"
          @hidden_layers << l
        end
      end
      net.layers.flatten.each do |layer|
        layer.neurons.each do |n|
          n.synapses_in.each do |s|
            source = @all_neurons.find { |i| i.id == s.source }
            destination = @all_neurons.find { |i| i.id == s.destination }
            next unless source && destination
            _s = Synapse.new(source, destination)
            _s.weight = s.weight
            source.synapses_out << _s
            destination.synapses_in << _s
            @all_synapses << _s
          end
          n.synapses_out.each do |s|
            source = @all_neurons.find { |i| i.id == s.source }
            destination = @all_neurons.find { |i| i.id == s.destination }
            next unless source && destination
            _s = Synapse.new(source, destination)
            _s.weight = s.weight
            source.synapses_in << _s
            destination.synapses_out << _s
            @all_synapses << _s
          end
        end
      end
      @logger.info("Network loaded from: #{file_path}")
    end

    def inspect
      @logger.info(@input_layers)
      @logger.info("--------------------------------")
      @logger.info(@hidden_layers)
      @logger.info("--------------------------------")
      @logger.info(@output_layers)
      @logger.info("--------------------------------")
      @logger.info(@all_synapses)
      @logger.info("--------------------------------")
    end
  end
end
