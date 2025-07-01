module SHAInet
  class RecurrentLayer < Layer
    property hidden_state

    def initialize(n_type : String, l_size : Int32, activation_function : ActivationFunction = SHAInet.sigmoid)
      super(n_type, l_size, activation_function)
      @hidden_state = Array(Float64).new(l_size, 0.0)
    end

    def reset_state
      @hidden_state.map! { 0.0 }
    end

    def activate_step
      new_state = Array(Float64).new(@l_size, 0.0)
      @hidden_state = new_state.clone
      new_state
    end

    def activate_sequence(sequence : Array(Array(GenNum)))
      outputs = [] of Array(Float64)
      sequence.each do |_|
        outputs << activate_step
      end
      outputs
    end
  end
end
