require "logger"

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
    getter :mean_error, total_error : Float64, w_gradient : Array(Float64), b_gradient : Array(Float64)

    # Parameters for SGD + Momentum
    property learning_rate : Float64, momentum : Float64

    # Parameters for Rprop
    property etah_plus : Float64, etah_minus : Float64, delta_max : Float64, delta_min : Float64
    getter prev_total_error : Float64

    # First creates an empty shell of the entire network
    def initialize(@logger : Logger = Logger.new(STDOUT))
      @input_layers = Array(Layer).new
      @output_layers = Array(Layer).new
      @hidden_layers = Array(Layer).new
      @all_neurons = Array(Neuron).new   # Array of all current neurons in the network
      @all_synapses = Array(Synapse).new # Array of all current synapses in the network

      @mean_error = Float64.new(1)     # Average netwrok error based on all the training so far
      @total_error = Float64.new(1)    # Sum of errors from output layer, based on a specific input
      @w_gradient = Array(Float64).new # Needed for batch train
      @b_gradient = Array(Float64).new # Needed for batch train

      @learning_rate = 0.7 # Standard parameter for GD
      @momentum = 0.3      # Improved GD

      @etah_plus = 1.2                          # For iRprop+ , how to increase step size
      @etah_minus = 0.5                         # For iRprop+ , how to decrease step size
      @delta_max = 50.0                         # For iRprop+ , max step size
      @delta_min = 0.1                          # For iRprop+ , min step size
      @prev_total_error = rand(0.0..1.0).to_f64 # For iRprop+ , needed for backtracking
    end

    # Create and populate a layer with neurons
    # l_type is: :input, :hidden or :output
    # l_size = how many neurons in the layer
    # n_type = advanced option for different neuron types
    def add_layer(l_type : Symbol, l_size : Int32, n_type : Symbol = :memory)
      layer = Layer.new(n_type, l_size, @logger)
      layer.neurons.each { |neuron| @all_neurons << neuron } # To easily access neurons later

      case l_type
      when :input
        @input_layers << layer
      when :hidden
        @hidden_layers << layer
      when :output
        @output_layers << layer
      else
        raise NeuralNetRunError.new("Must define correct layer type (:input, :hidden, :output).")
      end
    end

    # Connect all the layers in order (input and output don't connect between themselves): input, hidden, output
    def fully_connect
      # Connect all input layers to the first hidden layer
      @input_layers.each do |layer|
        layer.neurons.each do |neuron1|                  # Source neuron
          @hidden_layers.first.neurons.each do |neuron2| # Destination neuron
            synapse = Synapse.new(neuron1, neuron2)
            neuron1.synapses_out << synapse
            neuron2.synapses_in << synapse
            @all_synapses << synapse # To easily access synapes later
          end
        end
      end

      # Connect all hidden layer between each other hierarchically
      (0..@hidden_layers.size - 2).each do |l|
        @hidden_layers[l].neurons.each do |neuron1|       # Source neuron
          @hidden_layers[l + 1].neurons.each do |neuron2| # Destination neuron
            synapse = Synapse.new(neuron1, neuron2)
            neuron1.synapses_out << synapse
            neuron2.synapses_in << synapse
            @all_synapses << synapse
          end
        end
      end

      # Connect last hidden layer to all output layers
      @hidden_layers.last.neurons.each do |neuron1| # Source neuron
        @output_layers.each do |layer|
          layer.neurons.each do |neuron2| # Destination neuron
            synapse = Synapse.new(neuron1, neuron2)
            neuron1.synapses_out << synapse
            neuron2.synapses_in << synapse
            @all_synapses << synapse
          end
        end
      end
      @all_synapses.uniq
    end

    # Connect two specific layers with synapses
    def connect_ltl(layer1 : Layer, layer2 : Layer, connection_type : Symbol)
      raise NeuralNetInitalizationError.new("Error initilizing network, must choose correct connection type.") if CONNECTION_TYPES.any? { |x| x == connection_type } == false
      case connection_type
      # Connect each neuron from source layer to all neurons in destination layer
      when :full
        layer1.each do |neuron1|   # Source neuron
          layer2.each do |neuron2| # Destination neuron
            synapse = Synapse.new(neuron1, neuron2)
            neuron1.synapses_out << synapse
            neuron2.synapses_in << synapse
            @all_synapses << synapse
          end
        end
        # Connect each neuron from source layer to neuron with corresponding index in destination layer
      when :ind_to_ind
        raise NeuralNetInitalizationError.new("Error initializing network, index to index connection requires layers of same size.") if layer1.size != layer2.size
        (0..layer1.size).each do |index|
          synapse = Synapse.new(layer1[index], layer2[index])
          layer1[index].synapses_out << synapse
          layer2[index].synapses_in << synapse
          @all_synapses << synapse
        end

        # Randomly decide if each neuron from source layer will connect to a neuron from destination layer
      when :random
        layer1.each do |neuron1|   # Source neuron
          layer2.each do |neuron2| # Destination neuron
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
    end

    # Run an input throught the network to get an output (weights & biases do not change)
    def run(input : Array(GenNum), activation_function : Symbol = :sigmoid, stealth : Bool = false) : Array(Float64)
      raise NeuralNetRunError.new("Error input data doesn't fit input layers.") unless input.size == @input_layers.first.neurons.size

      # Insert the input data into the input layer
      @input_layers.first.neurons.each_with_index do |neuron, i| # Inserts the input information into the input layers
      # TODO: add support for multiple input layers
        neuron.activation = input[i].to_f64
      end

      # Propogate the information forward through the hidden layers
      @hidden_layers.each do |l|
        l.neurons.each { |neuron| neuron.activate(activation_function) }
      end

      # Propogate the information through the output layers
      @output_layers.each do |l|
        l.neurons.each { |neuron| neuron.activate(activation_function) }
      end

      output = @output_layers.last.neurons.map { |neuron| neuron.activation } # return an array of all output neuron activations
      # TODO: add support for multiple output layers

      unless stealth == true # Hide output report during training
        @logger.info("Input => #{input}, network output => #{output}")
      end
      output
    end

    # Quantifies how good the network performed for a single input compared to the expected output
    # This function returns the actual output and updates the error gradient for the output layer
    def evaluate(input : Array(GenNum), expected : Array(GenNum), cost_function : Symbol, activation_function : Symbol = :sigmoid)
      raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d).") if COST_FUNCTIONS.any? { |x| x == cost_function } == false

      actual = run(input, activation_function, stealth = true)
      raise NeuralNetRunError.new("Expected and actual output must be of the same dimention.") if expected.size != actual.size

      # Get the error signal for the final layer, based on the cost function (error signal is stored in the output neurons)
      total_error = [] of Float64
      case cost_function
      when :mse
        (0..expected.size - 1).each do |i|
          neuron = @output_layers.last.neurons[i] # Update error of all neurons in the output layer based on the actual result
          neuron.gradient = SHAInet.quadratic_cost_derivative(expected[i].to_f64, actual[i].to_f64)*neuron.sigma_prime
          # TODO: add support for multiple output layers
          total_error << SHAInet.quadratic_cost(expected[i].to_f64, actual[i].to_f64) # Store the output error based on cost function
          # @logger.info("i: #{i}, total_error array: #{total_error}")
        end
      when :c_ent
        (0..expected.size - 1).each do |i|
          neuron = @output_layers.last.neurons[i]
          neuron.gradient = SHAInet.cross_entropy_cost_derivative(expected[i].to_f64, actual[i].to_f64)*neuron.sigma_prime
          # TODO: add support for multiple output layers
          total_error << SHAInet.cross_entropy_cost(expected[i].to_f64, actual[i].to_f64)
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
      # @logger.info("total_error array: #{total_error}")
      total_error = total_error.reduce { |acc, i| acc + i } # Sum up all the errors from output layer
      # @logger.info("total_error reduce: #{total_error}")
      @total_error = total_error
    end

    # Online train, updates weights/biases after each data point (stochastic gradient descent)
    def train(data : Array(Array(Array(GenNum))), # Input structure: data = [[Input = [] of Float64],[Expected result = [] of Float64]]
              training_type : Symbol,             # Type of training: :sgdm, :rprop, :adam
              cost_function : Symbol,             # one of COST_FUNCTIONS described at the top of the file
              activation_function : Symbol,       # squashing performed on the activations within the network
              epochs : Int32,                     # a criteria of when to stop the training
              error_threshold : Float64,          # a criteria of when to stop the training
              log_each : Int32 = 1000)            # determines what is the step for error printout
      @logger.info("Training started")

      e = 0
      while e <= epochs
        all_errors = [] of Float64

        # Go over each data point and update the weights/biases based on the specific example
        data.each do |data_point|
          evaluate(data_point[0], data_point[1], cost_function, activation_function) # Update error gradient at the output layer based on current input
          all_errors << @total_error

          # Propogate the errors backwards through the hidden layers
          l = @hidden_layers.size - 1
          while l >= 0
            @hidden_layers[l].neurons.each { |neuron| neuron.hidden_error_prop } # Update neuron error based on errors*weights of neurons from the next layer
            l -= 1
          end

          @total_error = all_errors.reduce { |acc, i| acc + i }

          # Calculate MSE in %
          error_avg = @total_error/data.size
          sqrd_dist_sum = [] of Float64
          all_errors.each { |e| sqrd_dist_sum << (e - error_avg)**2 }

          @mean_error = 100*(sqrd_dist_sum.reduce { |acc, i| acc + i })/data.size

          # Update all wieghts & biases
          update_weights(training_type, batch = false)
          update_biases(training_type, batch = false)

          @prev_total_error = @total_error
        end

        if e % log_each == 0
          # @logger.info("Epoch: #{e}, error_sum: #{error_sum}")
          @logger.info("Epoch: #{e}, Total error: #{@total_error}, MSE: #{@mean_error}")
        end
        if @total_error >= error_threshold
          e += 1
        else
          @logger.info("Epoch: #{e}, Total error: #{@total_error}, MSE: #{@mean_error}")
          e += epochs
        end
      end
    end

    # Batch train, updates weights/biases using a gradient sum from all data points in the batch (using gradient descent)
    def train_batch(data : Array(Array(Array(GenNum))), # Input structure: data = [[Input = [] of Float64],[Expected result = [] of Float64]]
                    training_type : Symbol,             # Type of training: :sgdm, :rprop, :adam
                    cost_function : Symbol,             # one of COST_FUNCTIONS described at the top of the file
                    activation_function : Symbol,       # squashing performed on the activations within the network
                    epochs : Int32,                     # a criteria of when to stop the training
                    error_threshold : Float64,          # a criteria of when to stop the training
                    log_each : Int32 = 1000)            # determines what is the step for error printout

      @logger.info("Training started")

      e = 0
      while e <= epochs
        all_errors = [] of Float64
        batch_w_grad = [] of Array(Float64) # Save gradients from entire batch before updating weights & biases
        batch_b_grad = [] of Array(Float64)

        # Go over each data point and collect gradients of weights/biases based on each specific example
        data.each do |data_point|
          evaluate(data_point[0], data_point[1], cost_function, activation_function) # Get error gradient from output layer based on current input
          all_errors << @total_error

          # Propogate the errors backwards through the hidden layers
          l = @hidden_layers.size - 1
          while l >= 0
            @hidden_layers[l].neurons.each { |neuron| neuron.hidden_error_prop } # Update neuron error based on errors*weights of neurons from the next layer
            l -= 1
          end

          # Save all gradients from each data point for the batch update
          w_grad = [] of Float64
          b_grad = [] of Float64

          @all_synapses.each { |synapse| w_grad << (synapse.source_neuron.activation)*(synapse.dest_neuron.gradient) }
          batch_w_grad << w_grad
          @all_neurons.each { |neuron| b_grad << neuron.gradient }
          batch_b_grad << b_grad
        end

        # Sum up gradients into a single array
        batch = batch_w_grad.transpose
        @w_gradient = [] of Float64
        batch.each { |array| @w_gradient << array.reduce { |acc, i| acc + i } }
        batch = batch_b_grad.transpose
        @b_gradient = [] of Float64
        batch.each { |array| @b_gradient << array.reduce { |acc, i| acc + i } }

        @total_error = all_errors.reduce { |acc, i| acc + i }

        # Calculate MSE in %
        error_avg = @total_error/data.size
        sqrd_dist_sum = [] of Float64
        all_errors.each { |e| sqrd_dist_sum << (e - error_avg)**2 }

        @mean_error = 100*(sqrd_dist_sum.reduce { |acc, i| acc + i })/data.size

        # Update all wieghts & biases
        update_weights(training_type, batch = true)
        update_biases(training_type, batch = true)

        # # prevent local minimum
        # if (0.999*@prev_total_error < @total_error < @prev_total_error*1.001) == true
        #   randomize_all_weights
        #   randomize_all_biases
        # end

        @prev_total_error = @total_error

        if e % log_each == 0
          @logger.info("Epoch: #{e}, Total error: #{@total_error}, MSE: #{@mean_error}")
        end
        if @total_error >= error_threshold
          e += 1
        else
          @logger.info("Epoch: #{e}, Total error: #{@total_error}, MSE: #{@mean_error}")
          e += epochs
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
        end
      end
    end

    # Update biases based on the gradients and delta rule (including momentum)
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
