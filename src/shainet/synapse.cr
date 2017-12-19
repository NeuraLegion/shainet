module SHAInet
  class Synapse
    property weight : Float64, :source_neuron, :dest_neuron

    def initialize(@source_neuron : Neuron, @dest_neuron : Neuron)
      @weight = rand(0.0..1.0)
    end

    def randomize_weight
      @weight = rand(0.0..1.0).to_f64
    end

    def update_weight(value : Float64) : Float64
      @weight = value
    end

    # Transfer memory from source_neuron to dest_neuron while applying weight
    def propagate_forward : Float64
      new_memory = @source_neuron.memory*@weight

      case @source_neuron.n_type
      when :memory
        return new_memory
      when :eraser
        return (-1)*new_memory
      else
        raise "Other types of neurons are not supported yet!"
      end
    end

    # Transfer error & bias from dest_neuron to source_neuron while applying weight
    def propagate_backward : Float64
      new_error = @dest_neuron.error*@weight
      # new_bias = @dest_neuron.bias*@weight # ## not sure this is correct
      return new_error # , new_bias
    end

    def inspect
      pp @weight
      pp @bias
      pp @source_neuron
      pp @dest_neuron
    end
  end
end
