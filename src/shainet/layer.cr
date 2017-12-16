module SHAInet
  LAYER_TYPES = [:memory, :eraser, :amplifier, :fader, :sensor, :mixed]

  class Layer
    property :neurons, :l_type, :memory_size

    def initialize(l_size : Int32, @l_type : Symbol, @memory_size : Int32)
      raise NeuralNetInitalizationError.new("Must define correct layer type, if you're not sure choose :memory as a default") if LAYER_TYPES.any? { |x| x == l_type } == false
      @neurons = Array(Neuron).new(l_size) { |i| Neuron.new(@l_type, @memory_size) }
    end

    # If you don't want neurons to have a blank memory of zeros
    def random_seed
      @neurons.each do |neuron|
        neuron.memory = Array(Float64).new(memory_size) { |i| rand(-1.0..1.0) }
      end
      puts "Layers seeded with random values"
    end

    # If you want to change the memory size of all neurons in a layer
    def memory_change(new_memory_size : Int32)
      @neurons.each do |neuron|
        neuron.memory = Array(Float64).new(new_memory_size) { |i| 0.0 }
      end
      puts "Memory size changed from #{@memory_size} to #{new_memory_size}"
      @memory_size = new_memory_size
    end

    # If you want to change the type of layer including all neuron types within it
    def type_change(new_layer_type : Symbol)
      raise NeuralNetInitalizationError.new("Must define correct layer type, if you're not sure choose :memory as a default") if LAYER_TYPES.any? { |x| x == l_type } == false
      @neurons.each { |neuron| neuron.n_type = new_layer_type }
      puts "Layer type chaged from #{@l_type} to #{new_layer_type}"
      @l_type = new_layer_type
    end
  end
end
