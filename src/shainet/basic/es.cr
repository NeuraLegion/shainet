# Note: this type of optimization was inspired by the following paper:
# https://blog.openai.com/evolution-strategies/

module SHAInet
  class Pool
    getter organisms : Array(Organism), pool_biases : Array(Float64), pool_weights : Array(Float64)

    # property mvp : Organism

    def initialize(@network : Network,
                   @pool_size : Int32,
                   @learning_rate : Float64,
                   @sigma : Float64)
      #
      raise "Pool size must be at least 2" if @pool_size < 2
      raise "Sigma (std dev for sampling) must be > 0" if @sigma <= 0.0

      # Store previous data to avoid moving towards worse network states
      @pool_biases = Array(Float64).new
      @network.all_neurons.each { |neuron| @pool_biases << neuron.bias }

      @pool_weights = Array(Float64).new
      @network.all_synapses.each { |synapse| @pool_weights << synapse.weight }

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
      # @mvp = @organisms.sample.as(Organism)
    end

    def normalize_rewards
      reward_mean = 0.0
      reward_stdv = 0.0

      # Calculate mean
      @organisms.each do |organism|
        reward_mean += organism.reward
      end
      reward_mean /= @organisms.size

      # puts "reward_mean: #{reward_mean}"

      # Calculate standard deviation
      @organisms.each do |organism|
        reward_stdv += ((organism.reward - reward_mean)**2 / (@organisms.size - 1))**0.5
      end

      @organisms.each do |organism|
        # puts "reward_stdv: #{reward_stdv}"
        old_reward = organism.reward.clone
        organism.reward = (old_reward - reward_mean) / reward_stdv
        if organism.reward.nan?
          organism.reward = (old_reward - reward_mean)
        end
        # puts "organism.reward: #{organism.reward}"
      end
    end

    def pull_params
      normalize_rewards
      norm_value = @learning_rate / (@pool_size * @sigma)
      # puts "norm_value: #{norm_value}"

      # Sum the relative change each organism provides
      @organisms.each do |organism|
        organism.biases.each_with_index do |bias, i|
          weighted_value = bias * organism.reward
          @pool_biases[i] += norm_value * weighted_value
          # puts "i: #{i}"
          # puts "organism.error_signal: #{organism.error_signal}"
          # puts "organism.reward: #{organism.reward}"
          # puts "bias: #{bias}"
          # puts "weighted_value: #{weighted_value}"
        end

        organism.weights.each_with_index do |weight, i|
          weighted_value = weight * organism.reward
          @pool_weights[i] += norm_value * weighted_value
        end
      end

      # puts "@pool_biases: #{@pool_biases}"

      # Update network biases
      @network.all_neurons.each_with_index do |neuron, i|
        neuron.bias = @pool_biases[i].clone
      end

      # Update network weights
      @network.all_synapses.each_with_index do |synapse, i|
        synapse.weight = @pool_weights[i].clone
      end
    end
  end

  class Organism
    property reward : Float64
    getter mse : Float64, error_signal : Array(Float64)
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
      @mse = 0.0
      @reward = 0.0
      @error_signal = [] of Float64

      @original_biases = original_biases.clone
      @original_weights = original_weights.clone
      @biases = original_biases.clone
      @weights = original_weights.clone
    end

    def get_new_params
      # Update biases
      @network.all_neurons.each_with_index do |neuron, i|
        new_value = SHAInet::RandomNormal.sample(n: 1, mu: neuron.bias, sigma: @sigma).first
        neuron.bias = new_value.clone
        @biases[i] = new_value.clone
      end

      # Update weights
      @network.all_synapses.each_with_index do |synapse, i|
        new_value = SHAInet::RandomNormal.sample(n: 1, mu: synapse.weight, sigma: @sigma).first
        synapse.weight = new_value
        @weights[i] = new_value
      end
    end

    def update_reward
      @error_signal = @network.error_signal.clone
      @reward = 0.0
      @error_signal.each { |v| @reward -= v }

      # puts "###############"
      # puts "@error_signal: #{@error_signal}"
      # puts "@reward: #{@reward}"
      # puts "###############"
    end
  end
end
