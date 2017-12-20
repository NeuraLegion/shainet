require "logger"

module SHAInet
  class Network
    LAYER_TYPES      = [:input, :hidden, :output]
    CONNECTION_TYPES = [:full, :ind_to_ind, :random]
    COST_FUNCTIONS   = [:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d]

    property :input_layers, :output_layers, :hidden_layers, :all_neurons, :all_synapses
    property :activations, :input_sums, :biases, :weights, :mean_error
    property :error_signal, :error_gradient, :weight_gradient, :bias_gradient

    # First creates an empty shell of the entire network

    def initialize(@logger : Logger = Logger.new(STDOUT))
      @input_layers = Array(Layer).new
      @output_layers = Array(Layer).new
      @hidden_layers = Array(Layer).new
      @all_neurons = Array(Neuron).new   # Array of all current neurons in the network
      @all_synapses = Array(Synapse).new # Array of all current synapses in the network

      @activations = Array(Array(Float64)).new    # Matrix of activations (a), vector per layer
      @input_sums = Array(Array(Float64)).new     # Matrix of input sums (z), vector per layer
      @biases = Array(Array(Float64)).new         # Matrix of biases (b), vector per layer
      @weights = Array(Array(Array(Float64))).new # Array of weight matrices from each layer
      @mean_error = Float64.new(1)                # Average netwrok error based on all the training so far

      @error_signal = Array(Float64).new    # Array of errors for each neuron of the output layer (deltas)
      @error_gradient = Array(Float64).new  # Matrix of erross for each neuron in the hidden layers (deltas), vector per layer
      @weight_gradient = Array(Float64).new # Array of all individual slopes of weights based on the cost function (dC/dw)
      @bias_gradient = Array(Float64).new   # Array of all individual slopes of bias based on the cost function (dC/db)
    end

    # Create and populate a layer with neurons
    # l_type is: :input, :hidden or :output
    # l_size = how many neurons in the layer
    # n_type = advanced option for different neuron types
    def add_layer(l_type : Symbol, l_size : Int32, n_type : Symbol = :memory)
      layer = Layer.new(n_type, l_size, @logger)
      layer.neurons.each { |neuron| @all_neurons << neuron } # To easily access neurons later
      unless l_type == :input
        b_l = [] of Float64
        layer.neurons.each { |neuron| b_l << neuron.bias } # Vector of biases for the layer
        @biases << b_l                                     # Save all vector biases in a matrix (array of arrays)
      end
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

      # Save weights matrix for later
      @hidden_layers.each do |layer|
        w_l = [] of Array(Float64)
        layer.neurons.each do |neuron|
          w_n = [] of Float64
          neuron.synapses_in.each { |syn| w_n << syn.weight }
          w_l << w_n
        end
        @weights << w_l
      end
      @output_layers.each do |layer|
        w_l = [] of Array(Float64)
        layer.neurons.each do |neuron|
          w_n = [] of Float64
          neuron.synapses_in.each { |syn| w_n << syn.weight }
          w_l << w_n
        end
        @weights << w_l
      end
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
    def run(input : Array(Float64), activation_function : Symbol = :sigmoid) : Array(Float64)
      raise NeuralNetRunError.new("Error initializing network, input data doesn't fit input layers.") unless input.size == @input_layers.first.neurons.size

      @input_layers.first.neurons.each_with_index do |neuron, i| # Inserts the input information into the input layers
      # TODO: add support for multiple input layers
        neuron.activation = input[i]
      end

      @hidden_layers.each do |l| # Propogate the information through the hidden layers
        a_l = [] of Float64
        z_l = [] of Float64
        l.neurons.each do |neuron|
          neuron.activate(activation_function)
          a_l << neuron.activation
          z_l << neuron.input_sum
        end
        @activations << a_l # save activations vector for each layer
        @input_sums << z_l  # Save input sum vector for each layer
      end
      @output_layers.each do |l| # Propogate the information through the output layers
        a_l = [] of Float64
        z_l = [] of Float64
        l.neurons.each do |neuron|
          neuron.activate(activation_function)
          a_l << neuron.activation
        end
        @activations << a_l # save activations vector for each layer
        @input_sums << z_l  # Save input sum vector for each layer
      end
      output = @activations.last
      # TODO: add support for multiple output layers
      puts "For the input of #{input}, the networks output is: #{output}"
      return output
    end

    # Quantifies how good the network performed for a single input compared to the expected output
    # This function returns the actual output and updates the error gradient for the output layer
    def evaluate(input : Array(Float64), expected : Array(Float64), cost_function : Symbol, activation_function : Symbol = :sigmoid)
      raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d).") if COST_FUNCTIONS.any? { |x| x == cost_function } == false

      actual = run(input, activation_function)
      raise NeuralNetRunError.new("Expected and actual output must be of the same dimention.") if expected.size != actual.size

      @error_signal = Array(Float64).new
      case cost_function
      when :mse
        expected.size.times do |i|
          delta = SHAInet.quadratic_cost_derivative(expected[i], actual[i])*@output_layers.last.neurons[i].sigma_prime
          # TODO: add support for multiple output layers
          @error_signal << delta
        end
        return actual
      when :c_ent
        expected.size.times do |i|
          delta = SHAInet.cross_entropy_cost_derivative(expected[i], actual[i])*@output_layers.last.neurons[i].sigma_prime
          # TODO: add support for multiple output layers
          @error_signal << delta
        end
        return actual
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
    end

    # Input structure: data = [[Input = [] of Float64],[Expected result = [] of Float64]]
    # cost_function type is one of COST_FUNCTIONS described at the top of the file
    # epoch/error_threshold are criteria of when to stop the training
    # learning_rate is set to 0.3 only at the begining but will change dynamically with the total error, can be also changed manually
    def train_batch(data : Array(Array(Array(Float64))), cost_function : Symbol, epochs : Int32, error_threshold : Float64, learning_rate : Float64 = 0.3)
      puts "Training started\n----------"
      epochs.times do |i|
        all_errors = [] of Float64
        data.each do |data_point|                                                             # data_point = [input as array, expected output as array]
          actual = evaluate(data_point[0], data_point[1], cost_function, activation_function) # Get error gradiant from output layer based on current input
          all_errors << @error_signal.reduce { |acc, i| acc + i }                             # Save error from the last input

          @weight_gradient = [] of Float64 # Reset gradients
          @bias_gradient = [] of Float64

          # Propogate the errors backwards through the hidden layers
          l = @hidden_layers.size -1
          while l >= 0
            l_error_gradient = [] of Float64
            @hidden_layers[l].each do |neuron|
              neuron.error_prop                # Update neuron error based on errors*weights of neurons from the next layer
              l_error_gradient << neruon.error # Save error gradient of current leayer in an Array
            end
            @error_gradient << l_error_gradient # Save all error signals in a matrix
            @bias_gradient << l_error_gradient  #  bias gradient is equal to the layer gradient
            # @weight_gradient << SHAInet::dot_product(l_error_gradient,@activations[l-1].transpose)
            l -= 1
          end

          # Update weights & biases based on the error gradient

        end
        error_sum = all_errors.reduce { |acc, i| acc + i } # Sums all errors from last epoch
        @mean_error = error_sum/(data.size)
        puts "For epoch #{i}, MSE is #{@mean_error}\n----------"
      end
    end

    def train_batch(data : Array(Array(Float64)), epochs : Int32, error_threshold : Float64)
      # todo
    end

    # def update_weights(rate)
    #   synapses_in.each do |synapse|
    #     temp_weight = synapse.weight
    #     synapse.weight += (rate * LEARNING_RATE * error * synapse.source_neuron.output) + (MOMENTUM * (synapse.weight - synapse.prev_weight))
    #     synapse.prev_weight = temp_weight
    #   end
    #   temp_threshold = threshold
    #   @threshold += (rate * LEARNING_RATE * error * -1) + (MOMENTUM * (threshold - prev_threshold))
    #   @prev_threshold = temp_threshold
    # end

    def randomize_all_weights
      raise NeuralNetRunError.new("Cannot randomize weights without synapses") if @all_synapses.empty?
      @all_synapses.each &.randomize_weight
    end

    def randomize_all_biases
      raise NeuralNetRunError.new("Cannot randomize biases without synapses") if @all_synapses.empty?
      @all_synapses.each &.randomize_bias
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
