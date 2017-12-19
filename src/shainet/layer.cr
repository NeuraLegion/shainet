module SHAInet
  class Layer
    property :n_type, :neurons

    def initialize(@n_type : Symbol, l_size : Int32, @logger : Logger = Logger.new(STDOUT))
      @neurons = Array(Neuron).new
      # Populate layer with neurons
      l_size.times do
        @neurons << Neuron.new(@n_types)
      end
    end

    # If you don't want neurons to have a blank memory of zeros
    def random_seed
      @neurons.each do |neuron|
        neuron.memory = Array(Float64).new(memory_size) { |i| rand(-1.0..1.0) }
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
      @logger.info(@memory_size)
      @logger.info(@neurons)
    end
  end
end
