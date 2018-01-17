require "logger"

module SHAInet
  class SoftmaxLayer
    getter filters : Array(Array(Array(Array(Neuron)))), prev_layer : CNNLayer
    property next_layer : DummyLayer | CNNLayer | ConvLayer
    getter output : Array(Float64), :all_neurons

    # @softmax : Bool

    def initialize(@prev_layer : CNNLayer, @logger : Logger = Logger.new(STDOUT))
      unless @prev_layer.is_a?(SHAInet::FullyConnectedLayer)
        raise CNNInitializationError.new("Softmax layer can only follow a fully connected layer")
      end
      @filters = @prev_layer.filters.clone
      @output = Array(Float64).new(@filters[0][0][0].size) { Float64.new(0) }
      @all_neurons = Array(Neuron).new

      @next_layer = DummyLayer.new
      prev_layer.next_layer = self
    end

    def activate
      @output = Array(Float64).new(@filters[0][0][0].size) { Float64.new(0) }
      activations = Array(Float64).new
      @prev_layer.filters[0][0][0].size.times do |neuron|
        @prev_layer.filters[0][0][0][neuron].activate(@activation_function = SHAInet.none) # Activate previous neuron
        @filters[0][0][0][neuron] = @prev_layer.filters[0][0][0][neuron].clone             # Clone all information from previous neuron
        activations << @prev_layer.filters[0][0][0][neuron].activation
      end

      sf_activations = SHAInet.softmax(activations) # Calculate the softmax values based on entire output
      @filters[0][0][0].size.times do |neuron|      # Update each neuron's activation and derivative to fit the softmax values
        @filters[0][0][0][neuron].activation = sf_activations[neuron]
        @output[neuron] = sf_activations[neuron]
        @filters[0][0][0][neuron].sigma_prime = sf_activations[neuron]*(1 - sf_activations[neuron])
      end
    end

    def error_prop
      # Do nothing, this layer has to be last
    end

    def inspect(what : String)
      puts "Softmax layer:"
      case what
      when "weights"
        puts "Softmax layer has no weights"
      when "bias"
        puts "Softmax layer has no weights"
      when "activations"
        @filters.first.first.each { |row| puts "#{row.map { |n| n.activation.round(4) }}" }
      end
      puts "------------"
    end
  end
end
