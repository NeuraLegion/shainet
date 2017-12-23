module SHAInet
  class Synapse
    property weight : Float64, prev_weight : Float64 # , error : Float64
    getter source_neuron : Neuron, dest_neuron : Neuron

    def initialize(@source_neuron : Neuron, @dest_neuron : Neuron)
      @weight = rand(0.0..1.0).to_f64      # Weight of the synapse
      @prev_weight = rand(0.0..1.0).to_f64 # Needed for delta rule improvement (with momentum)
      # @error = rand(0.0..1.0).to_f64       # Error of the synapse
    end

    def randomize_weight
      @weight = rand(0.0..1.0).to_f64
    end

    def update_weight(value : Float64) : Float64
      @weight = value
    end

    # Transfer memory from source_neuron to dest_neuron while applying weight
    def propagate_forward : Float64
      new_memory = @source_neuron.activation*@weight

      case @source_neuron.n_type
      when :memory
        return new_memory
      when :eraser
        return (-1)*new_memory
      else
        raise "Other types of neurons are not supported yet!"
      end
    end

    # Transfer error from dest_neuron to source_neuron while applying weight and save the synapse gradient
    def propagate_backward : Float64
      weighted_error = @dest_neuron.error*@weight
      return weighted_error
    end

    def inspect
      pp @weight
      pp @source_neuron
      pp @dest_neuron
    end
  end
end
