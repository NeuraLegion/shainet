require "logger"

module SHAInet
  class NeuralNet
    LAYER_TYPES      = [:input, :hidden, :output]
    CONNECTION_TYPES = [:full, :ind_to_ind, :random]
    property :input_layers, :output_layers, :hidden_layers, :all_synapses, :error

    # First creates an empty shell of the entire network

    # def initialize(input_layers : Int32, hidden_layers : Int32, output_layers : Int32)
    #   raise NeuralNetInitalizationError.new("Error initializing network, there must be at least one layer from each type") if [input_layers, output_layers, hidden_layers].any? { |x| x <= 0 } == true
    def initialize(@logger : Logger = Logger.new(STDOUT))
      @input_layers = Array(Layer).new
      @output_layers = Array(Layer).new
      @hidden_layers = Array(Layer).new
      @all_synapses = Array(Synapse).new
    end

    # Populate each layer with neurons, must choose the neurons types and memory size per neuron
    def add_layer(l_type : Symbol, n_type : Symbol, l_size : Int32, memory_size : Int32 = 1)
      layer = Layer.new(n_type, l_size, memory_size, @logger)
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

    def evaluate(input : Array(Float64)) : Array(Float64)
      raise NeuralNetRunError.new("Error initializing network, input data doesn't fit input layers.") unless input.size == @input_layers.first.neurons.size

      @input_layers.first.neurons.each_with_index do |neuron, i|
        neuron.memory = [input[i]]
      end

      @hidden_layers.each { |l| l.neurons.each &.learn }
      @output_layers.each { |l| l.neurons.each &.learn }
      @output_layers.last.neurons.map { |n| n.memory.max }
    end

    def train(data : Array(Float64), epochs : Int32)
      # TODO
    end

    def randomize_all_weights
      raise NeuralNetRunError.new("Cannot randomize weights without synapses") if @all_synapses.empty?
      @all_synapses.each &.randomize_weight
    end

    def randomize_all_biases
      raise NeuralNetRunError.new("Cannot randomize bias without synapses") if @all_synapses.empty?
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
