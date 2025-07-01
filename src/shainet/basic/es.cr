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
      # Use matrix-based parameters instead of neurons
      all_layers = [@network.input_layers, @network.hidden_layers, @network.output_layers].flatten
      all_layers.each do |layer|
        if layer.biases
          layer.biases.rows.times do |r|
            layer.biases.cols.times do |c|
              @pool_biases << layer.biases[r, c]
            end
          end
        end
      end

      @pool_weights = Array(Float64).new
      # Use matrix-based weights instead of synapses
      all_layers.each do |layer|
        if layer.weights
          layer.weights.rows.times do |r|
            layer.weights.cols.times do |c|
              @pool_weights << layer.weights[r, c]
            end
          end
        end
      end

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
      bias_index = 0
      all_layers = [@network.input_layers, @network.hidden_layers, @network.output_layers].flatten
      all_layers.each do |layer|
        if layer.biases
          layer.biases.rows.times do |r|
            layer.biases.cols.times do |c|
              layer.biases[r, c] = @pool_biases[bias_index].clone
              bias_index += 1
            end
          end
        end
      end

      # Update network weights
      weight_index = 0
      all_layers.each do |layer|
        if layer.weights
          layer.weights.rows.times do |r|
            layer.weights.cols.times do |c|
              layer.weights[r, c] = @pool_weights[weight_index].clone
              weight_index += 1
            end
          end
        end
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
      bias_index = 0
      all_layers = [@network.input_layers, @network.hidden_layers, @network.output_layers].flatten
      all_layers.each do |layer|
        if layer.biases
          layer.biases.rows.times do |r|
            layer.biases.cols.times do |c|
              current_bias = layer.biases[r, c]
              new_value = SHAInet::RandomNormal.sample(n: 1, mu: current_bias, sigma: @sigma).first
              layer.biases[r, c] = new_value.clone
              @biases[bias_index] = new_value.clone
              bias_index += 1
            end
          end
        end
      end

      # Update weights
      weight_index = 0
      all_layers.each do |layer|
        if layer.weights
          layer.weights.rows.times do |r|
            layer.weights.cols.times do |c|
              current_weight = layer.weights[r, c]
              new_value = SHAInet::RandomNormal.sample(n: 1, mu: current_weight, sigma: @sigma).first
              layer.weights[r, c] = new_value
              @weights[weight_index] = new_value
              weight_index += 1
            end
          end
        end
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
