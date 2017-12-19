require "logger"

module SHAInet
  class Network
    LAYER_TYPES      = [:input, :hidden, :output]
    CONNECTION_TYPES = [:full, :ind_to_ind, :random]
    COST_FUNCTIONS   = [:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d]
    property :input_layers, :output_layers, :hidden_layers, :error_gradient, :all_weights, :all_biases, :weight_gradient, :bias_gradient, :mean_error

    # First creates an empty shell of the entire network

    # def initialize(input_layers : Int32, hidden_layers : Int32, output_layers : Int32)
    #   raise NeuralNetInitalizationError.new("Error initializing network, there must be at least one layer from each type") if [input_layers, output_layers, hidden_layers].any? { |x| x <= 0 } == true

    def initialize(@logger : Logger = Logger.new(STDOUT))
      @input_layers = Array(Layer).new
      @output_layers = Array(Layer).new
      @hidden_layers = Array(Layer).new
      @error_gradient = Array(Float64).new  # Array of errors for each neuron of the output layer
      @all_weights = Array(Float64).new     # Array of all current weights in the network
      @all_biases = Array(Float64).new      # Array of all current biases in the network
      @weight_gradient = Array(Float64).new # Array of all individual slopes of weights based on the cost function (dC/dw)
      @bias_gradient = Array(Float64).new   # Array of all individual slopes of bias based on the cost function (dC/db)
      @mean_error = Float64.new(1)
    end

    # Populate each layer with neurons, must choose the neurons types and memory size per neuron
    def add_layer(l_type : Symbol, l_size : Int32, n_type : Symbol = :memory)
      layer = Layer.new(n_type, l_size, @logger)
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
      (@input_layers.size - 1).times do |t|
        @input_layers[t].neurons.each do |neuron1|    # Source neuron
          @hidden_layers[0].neurons.each do |neuron2| # Destination neuron
            synapse = Synapse.new(neuron1, neuron2)
            neuron1.synapses_out << synapse
            neuron2.synapses_in << synapse
            @all_synapses << synapse
          end
        end
      end

      # Connect all hidden layer between each other hierarchically
      (@hidden_layers.size - 2).times do |t|
        @hidden_layers[t].neurons.each do |neuron1|       # Source neuron
          @hidden_layers[t + 1].neurons.each do |neuron2| # Destination neuron
            synapse = Synapse.new(neuron1, neuron2)
            neuron1.synapses_out << synapse
            neuron2.synapses_in << synapse
            @all_synapses << synapse
          end
        end
      end

      # Connect last hidden layer to all output layers
      @hidden_layers[-1].neurons.each do |neuron1| # Source neuron
        (@output_layers.size - 1).times do |t|
          @output_layers[t].neurons.each do |neuron2| # Destination neuron
            synapse = Synapse.new(neuron1, neuron2)
            neuron1.synapses_out << synapse
            neuron2.synapses_in << synapse
            @all_synapses << synapse
          end
        end
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
    def run(input : Array(Float64)) : Array(Float64)
      raise NeuralNetRunError.new("Error initializing network, input data doesn't fit input layers.") unless input.size == @input_layers.first.neurons.size

      @input_layers.first.neurons.each_with_index do |neuron, i| # Inserts the input information into the input layers
      # TODO: add support for multiple input layers
        neuron.memory = [input[i]]
      end

      @hidden_layers.each { |l| l.neurons.each &.activate } # Propogate the information through the hidden layers
      @output_layers.each { |l| l.neurons.each &.activate } # Propogate the information through the output layers
      @output_layers.last.neurons.map { |n| n.memory }      # Translate the output layer information to an array
      # TODO: add support for multiple output layers
    end

    # Quantifies how good the network performed for a single input compared to the expected output
    def evaluate(cost_function : Symbol, expected : Array(Float64), actual : Array(Float64)) : Float64
      raise NeuralNetRunError.new("Expected and actual output must be of the same dimention.") if expected.size != actual.size
      raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d).") if COST_FUNCTIONS.any? { |x| x == cost_function } == false

      case cost_function
      when :mse
        expected.size.times do |i| 
          @error_gradient << squared_cost(expected[i], actual[i]) }
      when :c_ent
        expected.size.times { |i| @error_gradient << cross_entropy_cost(expected[i], actual[i]) }
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
    def train(data : Array(Array(Array(Float64))), cost_function : Symbol, epochs : Int32, error_threshold : Float64, learning_rate : Float64 = 0.3)
      puts "Training started\n----------"
      epochs.each do |i|
        all_errors = Array(Float64).new
        data.size.times do |data_point|
          expected = data_point[1]                  # Array of expected Float64
          actual = evaluate(data_point[0])          # Array of float64 recieved as output from network
          evaluate(cost_function, expected, actual) # Get error gradiant from output layer based on current input
          @weight_gradient = Array(Float64).new
          @bias_gradient = Array(Float64).new
          l = @hidden_layers.size -1
          while l >= 0
            l_gradient = [] of Float64
            @hidden_layers[l].each do |neuron|
              neuron.error_prop          # Update neuron error based on errors*weights of neurons from the next layer
              l_gradient << neruon.error # Save error gradient of current leayer in an Array

              @weight_gradient
            end
            l -= 1
          end
        end
        error_sum = all_errors.reduce { |acc, i| acc + i } # Sums all errors from last epoch
        @mean_error = error_sum/(data.size)
        puts "For epoch #{i}, MSE is #{@mean_error}\n----------"
      end
    end

    def train_batch(data : Array(Array(Float64)), epochs : Int32, error_threshold : Float64)
      # todo
    end

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
