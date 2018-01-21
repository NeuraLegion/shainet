require "uuid"

module SHAInet
  # Each type of neuron uses and propogates data differently
  NEURON_TYPES = ["memory", "eraser", "amplifier", "fader", "sensor"]

  class Neuron
    property :synapses_in, :synapses_out, :n_type, activation : Float64, gradient : Float64, bias : Float64, prev_bias : Float64
    property input_sum : Float64, sigma_prime : Float64, gradient_sum : Float64, :id

    property prev_gradient : Float64, prev_delta : Float64, prev_delta_b : Float64
    property m_current : Float64, v_current : Float64, m_prev : Float64, v_prev : Float64

    def initialize(@n_type : String, @id : String = UUID.random.to_s)
      raise NeuralNetInitalizationError.new("Must choose currect neuron types, if you're not sure choose 'memory' as a standard neuron") unless NEURON_TYPES.includes?(@n_type)
      @synapses_in = [] of Synapse
      @synapses_out = [] of Synapse
      @activation = Float64.new(0)    # Activation of neuron after squashing function (a)
      @gradient = Float64.new(0)      # Error of the neuron, sometimes refered to as delta
      @bias = rand(-1..1).to_f64      # Activation threshhold (b)
      @prev_bias = rand(-1..1).to_f64 # Needed for delta rule improvement using momentum

      @input_sum = Float64.new(0)    # Sum of activations*weights from input neurons (z)
      @sigma_prime = Float64.new(1)  # derivative of input_sum based on activation function used (s')
      @gradient_sum = Float64.new(0) # Needed for batch train and back propagation of convolution layers

      # Parameters needed for Rprop
      @prev_gradient = rand(-0.1..0.1).to_f64
      @prev_delta = rand(0.0..0.1).to_f64
      @prev_delta_b = rand(-0.1..0.1).to_f64

      # Parameters needed for Adam
      @m_current = Float64.new(0) # Current moment value
      @v_current = Float64.new(0) # Current moment**2 value
      @m_prev = Float64.new(0)    # Previous moment value
      @v_prev = Float64.new(0)    # Previous moment**2 value
    end

    # This is the forward propogation
    # Allows the neuron to absorb the activation from its' own input neurons through the synapses
    # Then, it sums the information and an activation function is applied to normalize the data
    def activate(activation_function : Proc(GenNum, Array(Float64)) = SHAInet.sigmoid) : Float64
      new_memory = Array(Float64).new
      @synapses_in.each do |synapse| # Claclulate activation from each incoming neuron with applied weights, returns Array(Float64)
        new_memory << synapse.propagate_forward
      end
      @input_sum = new_memory.reduce { |acc, i| acc + i } # Sum all the information from input neurons, returns Float64
      @input_sum += @bias                                 # Add neuron bias (activation threshold)
      @activation, @sigma_prime = activation_function.call(@input_sum)
    end

    # This is the backward propogation of the hidden layers
    # Allows the neuron to absorb the error from its' own target neurons through the synapses
    # Then, it sums the information and a derivative of the activation function is applied to normalize the data
    def hidden_error_prop : Float64
      new_errors = [] of Float64
      @synapses_out.each do |synapse| # Calculate weighted error from each target neuron, returns Array(Float64)
        new_errors << synapse.propagate_backward
      end
      weighted_error_sum = new_errors.reduce { |acc, i| acc + i } # Sum weighted error from target neurons (instead of using w_matrix*delta), returns Float64
      @gradient = weighted_error_sum*@sigma_prime                 # New error of the neuron
    end

    def clone
      neuron_old = self
      neuron_new = Neuron.new(neuron_old.n_type)

      neuron_new.synapses_in = neuron_old.synapses_in.clone
      neuron_new.synapses_in.each { |synapse| synapse.dest_neuron = neuron_new }

      neuron_new.synapses_out = neuron_old.synapses_out.clone
      neuron_new.synapses_out.each { |synapse| synapse.source_neuron = neuron_new }

      neuron_new.activation = neuron_old.activation
      neuron_new.gradient = neuron_old.gradient
      neuron_new.bias = neuron_old.bias
      neuron_new.prev_bias = neuron_old.prev_bias
      neuron_new.input_sum = neuron_old.input_sum
      neuron_new.sigma_prime = neuron_old.sigma_prime
      neuron_new.prev_gradient = neuron_old.prev_gradient
      neuron_new.prev_delta = neuron_old.prev_delta
      neuron_new.prev_delta_b = neuron_old.prev_delta_b
      neuron_new.m_current = neuron_old.m_current
      neuron_new.v_current = neuron_old.v_current
      neuron_new.m_prev = neuron_old.m_prev
      neuron_new.v_prev = neuron_old.v_prev

      return neuron_new
    end

    def inspect
      pp @n_type
      pp @activation
      pp @gradient
      pp @sigma_prime
      pp @synapses_in
      pp @synapses_out
    end

    def randomize_bias
      @bias = rand(-1.0..1.0).to_f64
    end

    def update_bias(value : Float64)
      @bias = value
    end
  end
end
