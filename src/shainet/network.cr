require "logger"

module SHAInet
  class Network
    LAYER_TYPES      = [:input, :hidden, :output]
    CONNECTION_TYPES = [:full, :ind_to_ind, :random]
    COST_FUNCTIONS   = [:mse, :c_ent, :exp, :hel_d, :kld, :gkld, :ita_sai_d]

    getter :input_layers, :output_layers, :hidden_layers, :all_neurons, :all_synapses, :mean_error
    property learning_rate : Float64, momentum : Float64

    # property :activations, :input_sums, :biases, :weights
    # property :error_signal, :error_gradient, :weight_gradient, :bias_gradient

    # # Notes:
    # # ------------
    # # There are no matrices in this implementation, instead the gradient values are stored in each neuron/synapse independently.
    # # When preforming propogation, all the math is done iteratively on each neuron/synapse locally.

    # First creates an empty shell of the entire network

    def initialize(@logger : Logger = Logger.new(STDOUT))
      @input_layers = Array(Layer).new
      @output_layers = Array(Layer).new
      @hidden_layers = Array(Layer).new
      @all_neurons = Array(Neuron).new   # Array of all current neurons in the network
      @all_synapses = Array(Synapse).new # Array of all current synapses in the network
      @mean_error = Float64.new(1)       # Average netwrok error based on all the training so far

      @learning_rate = 0.7
      @momentum = 0.3
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
    rescue e : Exception
      raise NeuralNetRunError.new("Error fully connecting network: #{e}")
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
      input.each_with_index do |data, i|
        # Inserts the input information into the input layers
        # TODO: add support for multiple input layers
        @input_layers.first.neurons[i].activation = data.to_f64
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

      unless stealth # Hide output report during training
        @logger.info("Input => #{input}, network output => #{output}")
      end
      output
    rescue e : Exception
      raise NeuralNetRunError.new("Error running on layers: #{e} #{e.backtrace}")
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
        expected.size.times do |i|
          neuron = @output_layers.last.neurons[i] # Update error of all neurons in the output layer based on the actual result
          neuron.error = SHAInet.quadratic_cost_derivative(expected[i].to_f64, actual[i].to_f64)*neuron.sigma_prime
          # TODO: add support for multiple output layers
          total_error << SHAInet.quadratic_cost(expected[i].to_f64, actual[i].to_f64) # Store the output error based on cost function
        end
      when :c_ent
        expected.size.times do |i|
          neuron = @output_layers.last.neurons[i]
          neuron.error = SHAInet.cross_entropy_cost_derivative(expected[i].to_f64, actual[i].to_f64)*neuron.sigma_prime
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
      total_error = total_error.reduce { |acc, i| acc + i } # Sum up all the errors from output layer
      return total_error
    rescue e : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{e}")
    end

    # Input structure: data = [[Input = [] of Float64],[Expected result = [] of Float64]]
    # cost_function type is one of COST_FUNCTIONS described at the top of the file
    # epoch/error_threshold are criteria of when to stop the training
    # learning_rate is set to 0.3 only at the begining but will change dynamically with the total error, can be also changed manually
    def train(data : Array(Array(Array(GenNum))),
              cost_function : Symbol,
              activation_function : Symbol,
              epochs : Int32,
              error_threshold : Float64,
              log_each : Int32 = 100)
      @logger.info("Training started")

      e = 0
      while e <= epochs
        all_errors = [] of Float64

        # Go over each data point and update the weights/biases based on the specific example
        data.each_with_index do |data_point, i|
          if data_point.size < 2
            raise NeuralNetRunError.new("Error in training, dataset index number: #{i} doesn't have output (less then 2 members in array)")
          end
          total_error = evaluate(data_point[0], data_point[1], cost_function, activation_function) # Get error gradiant from output layer based on current input
          all_errors << total_error

          # Propogate the errors backwards through the hidden layers
          l = @hidden_layers.size - 1
          while l >= 0
            @hidden_layers[l].neurons.each { |neuron| neuron.hidden_error_prop } # Update neuron error based on errors*weights of neurons from the next layer
            l -= 1
          end

          # Update all wieghts & biases
          update_weights
          update_biases
        end
        # Get an average error for the last epoch
        error_sum = all_errors.reduce { |acc, i| acc + i }
        @mean_error = 100*error_sum/(data.size)
        if e % log_each == 0
          @logger.info("Epoch: #{e}, MSE: #{@mean_error}")
        end
        if @mean_error >= error_threshold
          e += 1
        else
          e += epochs
        end
      end
    rescue e : Exception
      @logger.error("Error in training: #{e} #{e.backtrace}")
      raise e
    end

    # def train_batch(data : Array(Array(Array(GenNum))),
    #                 cost_function : Symbol,
    #                 activation_function : Symbol,
    #                 epochs : Int32,
    #                 error_threshold : Float64)
    #   puts "Training started\n----------"

    #   e = 0
    #   while e <= epochs
    #     all_errors = [] of Float64
    #     batch_w_grad = [] of Array(Float64) # Save gradients from entire batch before updating weights & biases
    #     batch_b_grad = [] of Array(Float64)

    #     # Go over each data point and update the weights/biases based on the specific example
    #     data.each do |data_point|
    #       total_error = evaluate(data_point[0], data_point[1], cost_function, activation_function) # Get error gradiant from output layer based on current input
    #       all_errors << total_error

    #       # Propogate the errors backwards through the hidden layers
    #       l = @hidden_layers.size - 1
    #       while l >= 0
    #         @hidden_layers[l].neurons.each { |neuron| neuron.hidden_error_prop } # Update neuron error based on errors*weights of neurons from the next layer
    #         l -= 1
    #       end

    #       # Save all gradients from each data point for the batch update
    #       w_grad = [] of Float64
    #       b_grad = [] of Float64

    #       @all_synapses.each { |synapse| w_grad << (synapse.source_neuron.activation)*(synapse.dest_neuron.error) }
    #       batch_w_grad << w_grad
    #       @all_neurons.each { |neuron| b_grad << neuron.error }
    #       batch_b_grad << b_grad
    #     end

    #     # Sum up gradients into a single array
    #     batch = batch_w_grad.transpose
    #     w_grad = [] of Float64
    #     batch.each { |array| w_grad << array.reduce { |acc, i| acc + i } }
    #     batch = batch_b_grad.transpose
    #     b_grad = [] of Float64
    #     batch.each { |array| b_grad << array.reduce { |acc, i| acc + i } }

    #     # Update all wieghts & biases
    #     update_weights
    #     update_biases

    #     pp all_errors
    #     # Get an average error for the last epoch
    #     error_sum = all_errors.reduce { |acc, i| acc + i }
    #     @mean_error = error_sum/(data.size)
    #     puts "For epoch #{e}, mean error is #{@mean_error}\n----------"
    #     if @mean_error >= error_threshold
    #       e += 1
    #     else
    #       e += epochs
    #     end
    #   end
    # end

    # Update weights based on the gradients and delta rule (including momentum)
    def update_weights
      @all_synapses.reverse_each do |synapse|
        delta_weight = (-1)*@learning_rate*(synapse.source_neuron.activation)*(synapse.dest_neuron.error) + @momentum*(synapse.weight - synapse.prev_weight)
        synapse.weight += delta_weight
        synapse.prev_weight = synapse.weight
      end
    end

    # Update biases based on the gradients and delta rule (including momentum)
    def update_biases
      @all_neurons.reverse_each do |neuron|
        delta_bias = (-1)*@learning_rate*(neuron.error) + @momentum*(neuron.bias - neuron.prev_bias)
        neuron.bias += delta_bias
        neuron.prev_bias = neuron.bias
      end
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
