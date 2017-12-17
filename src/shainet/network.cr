module SHAInet
  class NeuralNet
    LAYER_TYPES      = [:input, :hidden, :output]
    CONNECTION_TYPES = [:full, :ind_to_ind, :random]
    property :input_layers, :output_layers, :hidden_layers, :all_synapses, :error

    # First creates an empty shell of the entire network

    # def initialize(input_layers : Int32, hidden_layers : Int32, output_layers : Int32)
    #   raise NeuralNetInitalizationError.new("Error initializing network, there must be at least one layer from each type") if [input_layers, output_layers, hidden_layers].any? { |x| x <= 0 } == true
    def initialize # (input_layers : Int32, hidden_layers : Int32, output_layers : Int32)
      @input_layers = Array(Layer).new
      @output_layers = Array(Layer).new
      @hidden_layers = Array(Layer).new
      @all_synapses = Array(Synapse).new
    end

    # Populate each layer with neurons, must choose the neurons types and memory size per neuron
    def add_layer(l_type : Symbol, n_type : Symbol, l_size : Int32, memory_size : Int32 = 1)
      raise NeuralNetInitalizationError.new("Must define correct layer type (:input, :hidden, :output).") if LAYER_TYPES.any? { |x| x == l_type } == false
      case l_type
      when :input
        @input_layers << Layer.new(n_type, l_size, memory_size)
      when :hidden
        @hidden_layers << Layer.new(n_type, l_size, memory_size)
      when :output
        @output_layers << Layer.new(n_type, l_size, memory_size)
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
      # raise
      unless input.size == @input_layers.first.neurons.size
        puts "Input: #{input.size}"
        puts "input_layers: #{@input_layers.first.neurons}"
        raise NeuralNetInitalizationError.new("Error initializing network, input data doesn't fit input layers.")
      end
      @input_layers.first.neurons.each_with_index do |neuron, i|
        neuron.memory = [input[i]]
      end

      @hidden_layers.each { |l| l.neurons.each &.learn }
      @output_layers.each { |l| l.neurons.each &.learn }
      @output_layers.last.neurons.map { |n| n.memory.max }
    end

    def train(data : Array(Float64), epochs : Int32)
      asda
    end

    def randomize_all_weights
      @all_synapses.each &.randomize_weight
    end

    def randomize_all_biases
      @all_synapses.each &.randomize_bias
    end

    def inspect
      pp @input_layers
      puts "--------------------------------"
      pp @hidden_layers
      puts "--------------------------------"
      pp @output_layers
      puts "--------------------------------"
      pp @all_synapses
      puts "--------------------------------"
    end
  end
end
