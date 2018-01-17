require "logger"

module SHAInet
  class SoftmaxLayer
    getter filters : Array(Filter), prev_layer : FullyConnectedLayer
    getter output : Array(Float64), :all_neurons

    def initialize(@prev_layer : FullyConnectedLayer, @logger : Logger = Logger.new(STDOUT))
      #
      unless @prev_layer.is_a?(SHAInet::FullyConnectedLayer)
        raise CNNInitializationError.new("Softmax layer can only follow a fully connected layer")
      end

      @filters = @prev_layer.filters.clone
      @output = Array(Float64).new
      @all_neurons = Array(Neuron).new
    end

    def activate
      @output = Array(Float64).new(@filters.first.neurons.first.size) { Float64.new(0) }
      input_sums = Array(Float64).new
      @prev_layer.filters.first.neurons.first.each { |neuron| input_sums << neuron.input_sum }

      sf_activations = SHAInet.softmax(input_sums)        # Calculate the softmax values based on entire output array
      @filters.first.neurons.first.size.times do |neuron| # Update each neuron's activation and derivative to fit the softmax values
        @filters.first.neurons.first[neuron].activation = sf_activations[neuron]
        @output[neuron] = sf_activations[neuron]
        @filters.first.neurons.first[neuron].sigma_prime = sf_activations[neuron]*(1 - sf_activations[neuron])
      end
    end

    # Send the gradients from current layer backwards without weights
    def error_prop
      @filters.size.times do |filter|
        @filters[filter].neurons.size.times do |row|
          @filters[filter].neurons[row].size.times do |neuron|
            @prev_layer.filters[filter].neurons[row][neuron].gradient = @filters[filter].neurons[row][neuron].gradient
          end
        end
      end
    end

    def inspect(what : String)
      case what
      when "weights"
        puts "Softmax layer has no weights"
      when "bias"
        puts "Softmax layer has no weights"
      when "activations"
        @filters.each_with_index do |filter, f|
          puts "---"
          puts "Filter: #{f}, neuron activations are:"
          filter.neurons.each do |row|
            puts "#{row.map { |n| n.activation.round(4) }}"
          end
        end
      when "gradients"
        @filters.each_with_index do |filter, f|
          puts "---"
          puts "Filter: #{f}, neuron gradients are:"
          filter.neurons.each do |row|
            puts "#{row.map { |n| n.gradient.round(4) }}"
          end
        end
      end
      puts "------------------------------------------------"
    end
  end
end
