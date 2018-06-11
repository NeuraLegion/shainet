# Note: this type of optimization was inspired by the following paper:
# https://blog.openai.com/evolution-strategies/

module SHAInet
  class Pool
    property :mvp
    getter :organisms

    @organisms : Array(Organism)
    @mvp : Organism

    def initialize(@network : Network,
                   @pool_size : Int32,
                   @learning_rate : Float64,
                   @sigma : Float64)
      #
      raise "Pool size must be at least 2" if @pool_size < 2
      # Store previous data to avoid moving towards worse network states
      @pool_biases = Array(Float64).new
      @pool_weights = Array(Float64).new
      save_nn_params

      # @pool_biases = Array(Float64).new
      # @pool_weights = Array(Float64).new

      @organisms = Array(Organism).new
      @pool_size.times do
        @organisms << Organism.new(
          network: @network,
          learning_rate: @learning_rate,
          sigma: @sigma,
          original_biases: @pool_biases,
          original_weights: @pool_weights
        )
      end
      @mvp = @organisms.sample.as(Organism)
    end

    def save_nn_params
      # @pool_biases = Array(Float64).new
      # @pool_weights = Array(Float64).new
      @network.all_neurons.each { |neuron| @pool_biases << neuron.bias }
      @network.all_synapses.each { |synapse| @pool_weights << synapse.weight }
    end

    # def restore_nn_params
    #   @network.all_neurons.each_with_index { |neuron, i| neuron.bias = @pool_biases[i] }
    #   @network.all_synapses.each_with_index { |synapse, i| synapse.weight = @pool_weights[i] }
    # end

    def normalize_rewards
      reward_mean = 0.0
      reward_stdv = 0.0

      # Calculate mean
      @organisms.each do |organism|
        reward_mean += organism.reward
      end
      reward_mean /= @organisms.size

      # Calculate standard deviation
      @organisms.each do |organism|
        reward_stdv += Math.sqrt((organism.reward - reward_mean)**2 / (@organisms.size - 1))
      end

      @organisms.each do |organism|
        organism.reward = (organism.reward.clone - reward_mean) / reward_stdv
      end
    end

    def pull_params
      normalize_rewards
      norm_value = @learning_rate / (@pool_size * @sigma)
      # puts "norm_value: #{norm_value}"

      @organisms.each do |organism|
        organism.biases.each_with_index do |bias, i|
          weighted_value = bias * organism.reward
          @pool_biases[i] += weighted_value # norm_value * weighted_value
          # puts "i: #{i}"
          # puts "organism.error_signal: #{organism.error_signal}"
          # puts "organism.reward: #{organism.reward}"
          # puts "bias: #{bias}"
          # puts "weighted_value: #{weighted_value}"
        end
        organism.weights.each_with_index do |weight, i|
          weighted_value = weight * organism.reward
          @pool_weights[i] += weighted_value # norm_value * weighted_value
        end
      end

      # Update network biases
      @network.all_neurons.each_with_index do |neuron, i|
        neuron.bias = @pool_biases[i].clone
      end

      # Update network weights
      @network.all_synapses.each_with_index do |synapse, i|
        synapse.weight = @pool_weights[i].clone
      end
    end

    # def reset
    #   @organisms = Array(Organism).new
    #   @pool_size.times do
    #     @organisms << Organism.new(
    #       network: @network,
    #       learning_rate: @learning_rate,
    #       sigma: @sigma,
    #       original_biases: @original_biases,
    #       original_weights: @original_weights
    #     )
    #   end
    # end

    # def natural_select(error_threshold : Float64)
    #   @original_biases = @mvp.original_biases.clone
    #   @original_weights = @mvp.original_weights.clone

    #   @organisms.each do |organism|
    #     if organism.mse > error_threshold
    #       organism.reset(@original_biases, @original_weights)
    #     else
    #       organism.update_params(@original_biases, @original_weights)
    #     end
    #   end
    # end
  end

  class Organism
    property mse : Float64, error_signal : Array(Float64), reward : Float64
    getter biases : Array(Float64), weights : Array(Float64)

    @network : Network
    @learning_rate : Float64
    @sigma : Float64
    @original_biases : Array(Float64)
    @original_weights : Array(Float64)

    def initialize(@network : Network,
                   @learning_rate : Float64,
                   @sigma : Float64,
                   original_biases : Array(Float64),
                   original_weights : Array(Float64))
      #
      # @learning_rate = rand(0.0..1.0)
      # @sigma = rand(0.0..1.0)
      @mse = 0.0
      @reward = 0.0
      @error_signal = [] of Float64

      @original_biases = original_biases.clone
      @original_weights = original_weights.clone
      @biases = original_biases.clone
      @weights = original_weights.clone
    end

    # def reset(original_biases : Array(Float64), original_weights : Array(Float64))
    #   @learning_rate = rand(0.0..1.0)
    #   @sigma = rand(0.0..1.0)
    #   @mse = 100000.0
    #   @original_biases = original_biases
    #   @original_weights = original_weights
    # end

    # def update_params(original_biases : Array(Float64), original_weights : Array(Float64))
    #   @original_biases = original_biases
    #   @original_weights = original_weights
    # end

    def get_new_params
      # Update biases
      @network.all_neurons.each_with_index do |neuron, i|
        # Only change value if mutation is triggered
        # This alows for some of the values to remain between epochs

        # if rand(0.0..1.0) < @sigma
        # Update networks biases using the organisms specific parameters
        # threshold = (@learning_rate*@original_biases[i]).abs

        change = rand(-@sigma..@sigma) # Add noise
        new_value = @original_biases[i] + change*@learning_rate
        neuron.bias = new_value.clone
        @biases[i] = new_value.clone

        # end
      end

      # Update weights
      @network.all_synapses.each_with_index do |synapse, i|
        # Only change value if mutation is triggered
        # This alows for some of the values to remain between epochs
        # if rand(0.0..1.0) < @sigma
        # Update networks biases using the organisms specific parameters
        # threshold = (@learning_rate*@original_weights[i]).abs

        change = rand(-@sigma..@sigma) # Add noise

        # change = rand(-threshold..threshold)
        new_value = @original_weights[i] + change*@learning_rate
        synapse.weight = new_value
        @weights[i] = new_value
        # end
      end
    end

    # def pull_params
    #   # Update biases
    #   @network.all_neurons.each_with_index do |neuron, i|
    #     neuron.bias = @biases[i].clone
    #   end

    #   # Update weights
    #   @network.all_synapses.each_with_index do |synapse, i|
    #     synapse.weight = @weights[i].clone
    #   end
    # end

    def update_reward
      @error_signal = @network.error_signal.clone
      @reward = 0.0
      @error_signal.each { |v| @reward -= v }
      # reward_sum = -@error_signal.reduce(0.0) { |acc, i| acc + i }
      # @reward = SHAInet._tanh(reward_sum)

      # puts "@reward: #{@reward}"
      # puts "@error_signal: #{@error_signal}"
      # @reward = -@mse.clone # ((@network.prev_mse - @mse) / @network.prev_mse)
      # puts "###############"
      # puts "@network.prev_mse: #{@network.prev_mse}"
      # puts "@mse: #{@mse}"
      # puts "reward: #{@reward}"
      # puts "###############"
    end
  end
end
