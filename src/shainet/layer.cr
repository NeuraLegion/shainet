module SHAInet
  class Layer
    property :n_type, :neurons
    getter :activation_function

    def initialize(@n_type : Symbol, l_size : Int32, @activation_function : Proc(GenNum, Array(Float64)) = SHAInet.sigmoid, @logger : Logger = Logger.new(STDOUT))
      @neurons = Array(Neuron).new
      # Populate layer with neurons
      l_size.times do
        @neurons << Neuron.new(@n_type)
      end
    end

    # If you don't want neurons to have a blank memory of zeros
    def random_seed
      @neurons.each do |neuron|
        neuron.activation = rand(-1.0..1.0).to_f64
      end
      @logger.info("Layers seeded with random values")
    end

    # If you want to change the type of layer including all neuron types within it
    def type_change(new_neuron_type : Symbol)
      raise NeuralNetRunError.new("Must define correct neuron type, if you're not sure choose :memory as a default") if NEURON_TYPES.any? { |x| x == new_neuron_type } == false
      @neurons.each { |neuron| neuron.n_type = new_neuron_type }
      @logger.info("Layer type chaged from #{@n_type} to #{new_neuron_type}")
      @n_type = new_neuron_type
    end

    def inspect
      @logger.info(@n_type)
      @logger.info(@neurons)
    end
  end
end
