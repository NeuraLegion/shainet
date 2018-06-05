module SHAInet
  class Pool
    property :mvp
    getter :organisms

    @organisms : Array(Organism)
    @mvp : Organism

    def initialize(@network : Network, @pool_size : Int32)
      # Store previous data to avoid moving towards worse network states
      @original_biases = Array(Float64).new
      @original_weights = Array(Float64).new
      @organisms = Array(Organism).new
      save_nn_params
      @pool_size.times do
        @organisms << Organism.new(@network, @original_biases, @original_weights)
      end
      @mvp = @organisms.sample.as(Organism)
    end

    def save_nn_params
      @network.all_neurons.each { |neuron| @original_biases << neuron.bias }
      @network.all_synapses.each { |synapse| @original_weights << synapse.weight }
    end

    def restore_nn_params
      @network.all_neurons.each_with_index { |neuron, i| neuron.bias = @original_biases[i] }
      @network.all_synapses.each_with_index { |synapse, i| synapse.weight = @original_weights[i] }
    end

    def reset
      @organisms = Array(Organism).new
      @pool_size.times do
        @organisms << Organism.new(@network, @original_biases, @original_weights)
      end
    end

    def natural_select(error_threshold : Float64)
      @organisms.each do |organism|
        if organism.mse > error_threshold
          organism.reset
        end
      end
    end
  end

  class Organism
    property mse : Float64

    @network : Network
    @learning_rate : Float64
    @mutation_chance : Float64
    @biases : Array(Float64)
    @weights : Array(Float64)

    def initialize(@network : Network, original_biases : Array(Float64), original_weights : Array(Float64))
      @learning_rate = rand(0.0..1.0)
      @mutation_chance = rand(0.0..1.0)
      @mse = 100000.0

      @original_biases = original_biases
      @original_weights = original_weights
      @biases = original_biases.clone
      @weights = original_weights.clone
    end

    def reset
      @learning_rate = rand(0.0..1.0)
      @mutation_chance = rand(0.0..1.0)
      @mse = 100000.0
      @biases = @original_biases.clone
      @weights = @original_weights.clone
    end

    def get_new_params
      # Update biases
      @network.all_neurons.each_with_index do |neuron, i|
        # Only change value if mutation is triggered
        # This alows for some of the values to remain between epochs
        if rand(0.0..1.0) < @mutation_chance
          # Update networks biases using the organisms specific parameters
          threshold = (@learning_rate*@original_biases[i]).abs

          change = rand(-threshold..threshold)
          new_value = @original_biases[i] + change
          neuron.bias = new_value

          @biases[i] = new_value
        end
      end

      # Update weights
      @network.all_synapses.each_with_index do |synapse, i|
        # Only change value if mutation is triggered
        # This alows for some of the values to remain between epochs
        if rand(0.0..1.0) < @mutation_chance
          # Update networks biases using the organisms specific parameters
          threshold = (@learning_rate*@original_weights[i]).abs

          change = rand(-threshold..threshold)
          new_value = @original_weights[i] + change
          synapse.weight = new_value
          @weights[i] = new_value
        end
      end
    end

    def pull_params
      # Update biases
      @network.all_neurons.each_with_index do |neuron, i|
        neuron.bias = @biases[i].clone
      end

      # Update weights
      @network.all_synapses.each_with_index do |synapse, i|
        synapse.weight = @weights[i].clone
      end
    end
  end
end
