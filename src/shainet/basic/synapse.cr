module SHAInet
  class Synapse
    property source_neuron : Neuron, dest_neuron : Neuron
    property weight : Float64, gradient : Float64, gradient_sum : Float64, gradient_batch : Float64, prev_weight : Float64
    property prev_gradient : Float64, prev_delta : Float64, prev_delta_w : Float64
    property m_current : Float64, v_current : Float64, m_prev : Float64, v_prev : Float64

    def initialize(@source_neuron : Neuron, @dest_neuron : Neuron)
      @weight = rand(-0.1_f64..0.1_f64)   # Weight of the synapse
      @gradient = rand(-0.1_f64..0.1_f64) # Error of the synapse with respect to cost function (dC/dW)
      @gradient_sum = 0_f64               # Needed for batch train
      @gradient_batch = 0_f64             # Needed for batch train
      @prev_weight = 0_f64                # Needed for delta rule improvement (with momentum)

      # Parameters needed for Rprop
      @prev_gradient = 0.0_f64
      @prev_delta = 0.1_f64
      @prev_delta_w = 0.1_f64

      # Parameters needed for Adam
      @m_current = 0_f64 # Current moment value
      @v_current = 0_f64 # Current moment**2 value
      @m_prev = 0_f64    # Previous moment value
      @v_prev = 0_f64    # Previous moment**2 value
    end

    # Transfer memory from source_neuron to dest_neuron while applying weight
    def propagate_forward : Float64
      new_memory = @source_neuron.activation*@weight

      case @source_neuron.n_type
      when "memory"
        new_memory
      when "eraser"
        (-1)*new_memory
      else
        raise "Other types of neurons are not supported yet!"
      end
    end

    # Transfer error from dest_neuron to source_neuron while applying weight and save the synapse gradient
    def propagate_backward : Float64
      @dest_neuron.gradient*@weight # weighted_error
    end

    def randomize_weight
      @weight = rand(-0.1_f64..0.1_f64)
    end

    def clone
      synapse_old = self
      synapse_new = Synapse.new(synapse_old.source_neuron, synapse_old.dest_neuron)

      synapse_new.weight = synapse_old.weight
      synapse_new.gradient = synapse_old.gradient
      synapse_new.prev_weight = synapse_old.prev_weight
      synapse_new.prev_gradient = synapse_old.prev_gradient
      synapse_new.prev_delta = synapse_old.prev_delta
      synapse_new.prev_delta_w = synapse_old.prev_delta_w
      synapse_new.m_current = synapse_old.m_current
      synapse_new.v_current = synapse_old.v_current
      synapse_new.m_prev = synapse_old.m_prev
      synapse_new.v_prev = synapse_old.v_prev
      synapse_new
    end

    def inspect
      pp @weight
      pp @source_neuron
      pp @dest_neuron
    end

    # Methods for Pointer matrix implementation - experimental
    def prev_weight_ptr
      pointerof(@prev_weight)
    end

    def weight_ptr
      pointerof(@weight)
    end

    def gradient_ptr
      pointerof(@gradient)
    end
  end
end
