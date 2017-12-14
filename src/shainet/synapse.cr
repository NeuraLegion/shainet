module SHAInet
  class Synapse
    property :wight

    @wight : Float64

    def initialize(@master : Neuron, @slave : Neuron, @wight : Float64)
    end

    def update_wight(value : Float64)
      @wight = value
    end

    def randomize_wight
      @wight = rand(0.0..1.0).to_f64
    end

    def propagate
      # Transfer data from @master to @slave using @wight value
      @slave.update_matrix(@master.matrix, @wight)
    end
  end
end
