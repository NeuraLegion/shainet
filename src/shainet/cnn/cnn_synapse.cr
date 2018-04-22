require "logger"

module SHAInet
  class CnnSynapse
    property weight : Float64, gradient : Float64, gradient_sum : Float64, gradient_batch : Array(Float64), prev_weight : Float64
    property prev_gradient : Float64, prev_delta : Float64, prev_delta_w : Float64
    property m_current : Float64, v_current : Float64, m_prev : Float64, v_prev : Float64

    def initialize
      @weight = rand(-1.0..1.0).to_f64   # Weight of the synapse
      @gradient_sum = Float64.new(0)     # For backpropogation of ConvLayers
      @gradient = rand(-0.1..0.1).to_f64 # Error of the synapse with respect to cost function (dC/dW)
      @gradient_batch = Float64.new(0)   # For batch-train

      @prev_weight = Float64.new(0) # Needed for delta rule improvement (with momentum)

      # Parameters needed for Rprop
      @prev_gradient = 0.0
      @prev_delta = 0.1
      @prev_delta_w = 0.1

      # Parameters needed for Adam
      @m_current = Float64.new(0) # Current moment value
      @v_current = Float64.new(0) # Current moment**2 value
      @m_prev = Float64.new(0)    # Previous moment value
      @v_prev = Float64.new(0)    # Previous moment**2 value
    end

    def randomize_weight
      @weight = rand(-0.1..0.1).to_f64
    end

    def clone
      synapse_old = self
      synapse_new = CnnSynapse.new

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
      return synapse_new
    end

    def inspect
      pp @weight
      pp @source_neuron
      pp @dest_neuron
    end
  end
end
