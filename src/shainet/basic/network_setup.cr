require "logger"
require "json"

module SHAInet
  class Network
    include JSON::Serializable
    # Notes:
    # ------------
    # There are no matrices in this implementation, instead the gradient values
    # are stored in each neuron/synapse independently.
    # When preforming propogation,
    # all the math is done iteratively on each neuron/synapse locally.
    #
    # This file contains all the methods for creating and maintaining
    # the network, for methods regarding running and training go to network_run.cr
    # ------------

    LAYER_TYPES      = ["input", "hidden", "output"]
    CONNECTION_TYPES = ["full", "ind_to_ind", "random"]
    COST_FUNCTIONS   = ["mse", "c_ent"] # , "exp", "hel_d", "kld", "gkld", "ita_sai_d"]

    # General network parameters
    getter :input_layers, :output_layers, :hidden_layers, :all_neurons, :all_synapses
    getter error_signal : Array(Float64), total_error : Float64, :mse, w_gradient : Array(Float64), b_gradient : Array(Float64)

    @[JSON::Field(ignore: true)]
    @logger : Logger = Logger.new(STDOUT)
    # Parameters for SGD + Momentum
    property learning_rate : Float64, momentum : Float64

    # Parameters for Rprop
    property etah_plus : Float64, etah_minus : Float64, delta_max : Float64, delta_min : Float64
    getter prev_mse : Float64

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
      @total_error = 1_f64               # Sum of errors from output layer, based on a specific input
      @mse = 1_f64                       # MSE of netwrok, based on all errors of output layer for a specific input or batch
      @w_gradient = Array(Float64).new   # Needed for batch train
      @b_gradient = Array(Float64).new   # Needed for batch train

      @learning_rate = 0.005_f64 # Standard parameter for GD
      @momentum = 0.05_f64       # Improved GD

      @etah_plus = 1.2_f64  # For iRprop+ , how to increase step size
      @etah_minus = 0.5_f64 # For iRprop+ , how to decrease step size
      @delta_max = 50_f64   # For iRprop+ , max step size
      @delta_min = 0.1_f64  # For iRprop+ , min step size
      @prev_mse = 1_f64     # For iRprop+ , needed for backtracking

      @alpha = 0.001_f64   # For Adam , step size (recomeneded: only change this hyper parameter when fine-tuning)
      @beta1 = 0.9_f64     # For Adam , exponential decay rate (not recommended to change value)
      @beta2 = 0.999_f64   # For Adam , exponential decay rate (not recommended to change value)
      @epsilon = 10e-8_f64 # For Adam , prevents exploding gradients (not recommended to change value)
      @time_step = 0_i32   # For Adam
    end

    # Create and populate a layer with neurons
    # l_type is: :input, :hidden or :output
    # l_size = how many neurons in the layer
    # n_type = advanced option for different neuron types
    def add_layer(l_type : Symbol | String, l_size : Int32, n_type : Symbol | String = "memory", activation_function : ActivationFunction = SHAInet.sigmoid)
      layer = Layer.new(n_type.to_s, l_size, activation_function, @logger)
      layer.neurons.each do |neuron|
        @all_neurons << neuron # To easily access neurons later
      end

      case l_type.to_s
      when "input"
        @input_layers << layer
      when "hidden"
        @hidden_layers << layer
      when "output"
        if @output_layers.empty?
          @output_layers << layer
        else
          @output_layers.delete(@output_layers.first)
          @output_layers << layer
          connect_ltl(@hidden_layers.last, @output_layers.first, :full)
        end
      else
        raise NeuralNetRunError.new("Must define correct layer type (:input, :hidden, :output).")
      end
    end

    # Connect all the layers in order (input and output don't connect between themselves): input, hidden, output
    def fully_connect
      if @hidden_layers.empty?
        # Connect all input layers to all output layers
        @output_layers.each do |out_layer|
          @input_layers.each do |in_layer|
            connect_ltl(in_layer, out_layer, :full)
          end
        end
      else
        # Connect all input layers to the first hidden layer
        @input_layers.each do |in_layer|
          connect_ltl(in_layer, @hidden_layers.first, :full)
        end

        # Connect all hidden layer between each other hierarchically
        (@hidden_layers.size).times do |l|
          next if (l + 1) == @hidden_layers.size
          connect_ltl(@hidden_layers[l], @hidden_layers[l + 1], :full)
        end

        # Connect last hidden layer to all output layers
        @output_layers.each do |out_layer|
          connect_ltl(@hidden_layers.last, out_layer, :full)
        end
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error fully connecting network: #{e}")
    end

    # Connect two specific layers with synapses
    def connect_ltl(src_layer : Layer, dest_layer : Layer, connection_type : Symbol | String)
      raise NeuralNetInitalizationError.new("Error initilizing network, must choose correct connection type.") if CONNECTION_TYPES.any? { |x| x == connection_type.to_s } == false
      case connection_type.to_s
      # Connect each neuron from source layer to all neurons in destination layer
      when "full"
        # Resize the weights matrix based on the connecting layer
        # dest_layer.weights.reshape_new(src_layer.size, dest_layer.size)

        src_layer.neurons.each_with_index do |src_neuron, src_i|
          dest_layer.neurons.each_with_index do |dest_neuron, dest_i|
            synapse = Synapse.new(src_neuron, dest_neuron)
            src_neuron.synapses_out << synapse
            dest_neuron.synapses_in << synapse
            @all_synapses << synapse

            # dest_layer.weights.data[dest_i][src_i] = synapse.weight_ptr

            # weights_vector << pointerof(synapse.weight)
            # prev_weights_vector << pointerof(synapse.prev_weight)
            # w_grad_vector << pointerof(synapse.gradient)
          end
        end
        # Connect each neuron from source layer to neuron with
        # corresponding index in destination layer
        # Matrix training is not implemented yet for this connection
      when "ind_to_ind"
        raise NeuralNetInitalizationError.new(
          "Error initializing network, index to index connection requires layers of same size.") if src_layer.neurons.size != dest_layer.neurons.size
        (0..src_layer.neurons.size).each do |index|
          synapse = Synapse.new(src_layer.neurons[index], dest_layer.neurons[index])
          src_layer.neurons[index].synapses_out << synapse
          dest_layer.neurons[index].synapses_in << synapse
          @all_synapses << synapse
        end

        # Randomly decide if each neuron from source layer will
        # connect to a neuron from destination layer
        # Matrix training is not implemented yet for this connection
      when "random"
        src_layer.neurons.each do |src_neuron|     # Source neuron
          dest_layer.neurons.each do |dest_neuron| # Destination neuron
            x = rand(0..1)
            if x <= 0.5 # Currently set to 50% chance, this can be changed at will
              synapse = Synapse.new(src_neuron, dest_neuron)
              src_neuron.synapses_out << synapse
              dest_neuron.synapses_in << synapse
              @all_synapses << synapse
            end
          end
        end
      end
      @all_synapses.uniq!
    rescue e : Exception
      raise NeuralNetRunError.new("Error in connect_ltl: #{e}")
    end

    def log_summary(e)
      @logger.info("Epoch: #{e}, Total error: #{@total_error}, MSE: #{@mse}")
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
        # elsif @hidden_layers.empty?
        #   raise NeuralNetRunError.new("Need atleast one hidden layer")
      elsif @output_layers.empty?
        raise NeuralNetRunError.new("No output layers defined")
      end
    end

    def randomize_all_weights
      raise NeuralNetRunError.new("Cannot randomize weights without synapses") if @all_synapses.empty?
      @all_synapses.each &.randomize_weight
    end

    def randomize_all_biases
      raise NeuralNetRunError.new("Cannot randomize biases without neurons") if @all_synapses.empty?
      @all_neurons.each &.randomize_bias
    end

    # File.write(self.to_json, path)

    # self.from_json(File.read(path))

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
