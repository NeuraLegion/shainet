require "logger"

module SHAInet
  class SoftmaxLayer
    getter filters : Array(Filter), prev_layer : FullyConnectedLayer | ReluLayer
    getter output : Array(Float64), :all_neurons

    def initialize(@prev_layer : FullyConnectedLayer | ReluLayer, @logger : Logger = Logger.new(STDOUT), @range : Range(Int32, Int32) = (0..-1))
      #
      @filters = @prev_layer.filters.clone

      @all_neurons = Array(Neuron).new
      @filters.first.neurons.first.each do |neuron|
        @all_neurons << neuron
      end

      begin
        @all_neurons[@range]
      rescue e : IndexError
        raise "Index is out of bounds, range surpasses neurons array size. Max size is #{@all_neurons.size}"
      end

      @output = Array(Float64).new(@all_neurons.size) { Float64.new(0) }
    end

    def activate
      # @filters.first.neurons.first.each_with_index do |neuron, i|
      @all_neurons.each_with_index do |neuron, i|
        neuron.activation = @prev_layer.filters.first.neurons.first[i].activation
        # @all_neurons << neuron
      end
      input_sums = Array(Float64).new

      # if @range[1] == -1
      @all_neurons[@range].each { |neuron| input_sums << neuron.activation }

      sf_activations = SHAInet.softmax(input_sums) # Calculate the softmax values based on entire output array
      # sf_activations = SHAInet.log_softmax(input_sums)    # Calculate the log softmax values based on entire output array

      @all_neurons[@range].each_with_index do |neuron, i| # Update each neuron's activation and derivative to fit the softmax values
        neuron.activation = sf_activations[i]
        neuron.sigma_prime = sf_activations[i]*(1 - sf_activations[i])
      end

      @all_neurons.each_with_index { |neuron, i| @output[i] = neuron.activation }
    end

    # Send the gradients from current layer backwards without weights
    def error_prop(batch : Bool = false)
      @filters.size.times do |filter|
        @filters[filter].neurons.size.times do |row|
          @filters[filter].neurons[row].size.times do |neuron|
            @prev_layer.filters[filter].neurons[row][neuron].gradient = @filters[filter].neurons[row][neuron].gradient
          end
        end
      end
    end

    def update_wb(learn_type : Symbol | String, batch : Bool = false)
      # Do nothing
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
          puts "Softmax layer, neuron activations are:"
          filter.neurons.each do |row|
            puts "#{row.map { |n| n.activation.round(4) }}"
          end
        end
      when "gradients"
        @filters.each_with_index do |filter, f|
          puts "---"
          puts "Softmax layer, neuron gradients are:"
          filter.neurons.each do |row|
            puts "#{row.map { |n| n.gradient.round(4) }}"
          end
        end
      end
      puts "------------------------------------------------"
    end
  end
end
