require "logger"

module SHAInet
  # In conv layers a filter is a separate unit within the layer, with special parameters
  class Filter
    getter input_surface : Array(Int32), window_size : Int32
    property neurons : Array(Array(Neuron)), receptive_field : ReceptiveField

    def initialize(@input_surface : Array(Int32), # expecting [width, height, channels]
                   @window_size : Int32)
      #
      @neurons = Array(Array(Neuron)).new(input_surface[1]) {
        Array(Neuron).new(input_surface[0]) { Neuron.new("memory") }
      }
      @receptive_field = ReceptiveField.new(window_size, input_surface[2])
    end

    def clone
      filter_old = self
      filter_new = Filter.new(filter_old.input_surface, filter_old.window_size)

      filter_new.neurons = filter_old.neurons.clone
      filter_new.receptive_field = filter_old.receptive_field.clone
      return filter_new
    end
  end

  # ########################################################################################################## #

  # This is somewhat similar to a synapse
  class ReceptiveField
    property weights : Array(Array(Array(Float64))), bias : Float64
    getter window_size : Int32, channels : Int32

    def initialize(@window_size : Int32, @channels : Int32)
      @weights = Array(Array(Array(Float64))).new(channels) {
        Array(Array(Float64)).new(@window_size) {
          Array(Float64).new(@window_size) { rand(0.0..1.0).to_f64 }
        }
      }
      @bias = rand(-1..1).to_f64
    end

    # Takes a small window from the input data (CxHxW) to preform feed forward
    # Propagate forward from CNNLayer
    def prpogate_forward(input_window : Array(Array(Array(Neuron))), target_neuron : Neuron)
      weighted_sum = Float64.new(0)
      @weights.size.times do |channel|
        @weights[channel].size.times do |row|
          @weights[channel][row].size.times do |col|
            weighted_sum += input_window[channel][row][col].activation * @weights[channel][row][col]
          end
        end
      end
      target_neuron.activation = weighted_sum + @bias
    end

    # Propagate forward from ConvLayer
    def prpogate_forward(input_window : Array(Array(Neuron)), target_neuron : Neuron)
      weighted_sum = Float64.new(0)
      @weights.first.size.times do |row|
        @weights.first[row].size.times do |col|
          weighted_sum += input_window[row][col].activation*@weights.first[row][col]
        end
      end
      target_neuron.activation = weighted_sum + @bias
    end

    # Propagate forward from CNNLayer
    def propagate_backward(window : Array(Array(Array(Neuron))), target_neuron : Neuron)
      @weights.size.times do |channel|
        @weights[channel].size.times do |row|
          @weights[channel][row].size.times do |col|
            window[channel][row][col].gradient = target_neuron.gradient*@weights[channel][row][col]
          end
        end
      end
    end

    # Propagate forward from ConvLayer
    def propagate_backward(window : Array(Array(Neuron)), target_neuron : Neuron)
      @weights[0][0].size.times do |row|
        @weights.[0][0][row].size.times do |col|
          weighted_sum += input_window[row][col].activation*@weights.first[row][col]
        end
      end
      target_neuron.activation = weighted_sum + @bias
    end

    def clone
      rf_old = self
      rf_new = ReceptiveField.new(rf_old.window_size, rf_old.channels)
      rf_new.weights = rf_old.weights
      rf_new.bias = rf_old.bias

      return rf_new
    end
  end

  # # Bellow is a better object oriented implementation - work in progress

  # property synapses : Array(Array(Array(CnnSynapse))), bias : Float64
  # getter window_size : Int32, channels : Int32

  # def initialize(@window_size : Int32, @channels : Int32)
  #   @synapses = Array(Array(Array(Synapse))).new(channels) {
  #     Array(Array(Synapse)).new(@window_size) {
  #       Array(Synapse).new(@window_size) { CnnSynapse.new }
  #     }
  #   }
  #   @bias = rand(-1..1).to_f64
  # end

  # # Takes a small window from the input data (CxHxW), and connects the neurons using CnnSynapse
  # # When previous layer is CNNLayer
  # def connect(input_window : Array(Array(Array(Neuron))), target_neuron : Neuron)
  #   @synapses.size.times do |channel|
  #     @synapses[channel].size.times do |row|
  #       @synapses[channel][row].size.times do |col|
  #         @synapses[channel][row][col].neuron_pairs << [input_window[channel][row][col], target_neuron]
  #       end
  #     end
  #   end
  # end

  # # Takes a small window from the input data (CxHxW), and connects the neurons using CnnSynapse
  # # When previous layer is ConvLayer
  # def connect(input_window : Array(Array(Neuron)), target_neuron : Neuron)
  #   @synapses.size.times do |channel|
  #     @synapses[channel].size.times do |row|
  #       @synapses[channel][row].size.times do |col|
  #         @synapses[channel][row][col].neuron_pairs << [input_window[row][col], target_neuron]
  #       end
  #     end
  #   end
  # end

  # def propagate_forward
  #   @synapse.each do |channel|
  #     channel.each do |row|
  #       row.each do |synapse|
  #         synapse.propagate_forward
  #       end
  #     end
  #   end
  # end

  # # Takes a small window from the input data (CxHxW) to preform feed forward
  # # Propagate forward from CNNLayer
  # def prpogate_forward(input_window : Array(Array(Array(Neuron))), target_neuron : Neuron)
  #   weighted_sum = Float64.new(0)
  #   @weights.each_with_index do |_c, channel|
  #     @weights[channel].each_with_index do |_r, row|
  #       @weights[channel][row].each_with_index do |_c, col|
  #         weighted_sum += input_window[channel][row][col].activation * @weights[channel][row][col]
  #       end
  #     end
  #   end
  #   target_neuron.activation = weighted_sum + @bias
  # end

  # # Propagate forward from ConvLayer
  # def prpogate_forward(input_window : Array(Array(Neuron)), target_neuron : Neuron)
  #   weighted_sum = Float64.new(0)
  #   @weights.first.size.times do |row|
  #     @weights.first[row].size.times do |col|
  #       weighted_sum += input_window[row][col].activation*@weights.first[row][col]
  #     end
  #   end
  #   target_neuron.activation = weighted_sum + @bias
  # end

  # class CnnSynapse
  #   property neuron_pairs : Array(Array(Neuron))
  #   property weight : Float64, gradient : Float64, prev_weight : Float64
  #   property prev_gradient : Float64, prev_delta : Float64, prev_delta_w : Float64
  #   property m_current : Float64, v_current : Float64, m_prev : Float64, v_prev : Float64

  #   def initialize(@neuron_pairs : Array(Array(Neuron)))
  #     @neuron_pairs = Array(Array(Neuron)).new # Array of pairs [source neuron, target neuron]
  #     @weight = rand(-1.0..1.0).to_f64         # Weight of the synapse
  #     @gradient = rand(-0.1..0.1).to_f64       # Error of the synapse with respect to cost function (dC/dW)
  #     @prev_weight = Float64.new(0)            # Needed for delta rule improvement (with momentum)

  #     # Parameters needed for Rprop
  #     @prev_gradient = 0.0
  #     @prev_delta = 0.1
  #     @prev_delta_w = 0.1

  #     # Parameters needed for Adam
  #     @m_current = Float64.new(0) # Current moment value
  #     @v_current = Float64.new(0) # Current moment**2 value
  #     @m_prev = Float64.new(0)    # Previous moment value
  #     @v_prev = Float64.new(0)    # Previous moment**2 value
  #   end

  #   # Transfer memory from source_neuron to dest_neuron while applying weight
  #   def propagate_forward : Float64
  #     @neuron_pairs.each do |pair|
  #       pair[1].activation
  #     @source_neuron.activation*@weight

  #   end

  #   # Transfer error from dest_neuron to source_neuron while applying weight and save the synapse gradient
  #   def propagate_backward : Float64
  #     weighted_error = @dest_neuron.gradient*@weight
  #     return weighted_error
  #   end

  #   def randomize_weight
  #     @weight = rand(-0.1..0.1).to_f64
  #   end

  #   def clone
  #     synapse_old = self
  #     synapse_new = Synapse.new(synapse_old.source_neuron, synapse_old.dest_neuron)

  #     synapse_new.weight = synapse_old.weight
  #     synapse_new.gradient = synapse_old.gradient
  #     synapse_new.prev_weight = synapse_old.prev_weight
  #     synapse_new.prev_gradient = synapse_old.prev_gradient
  #     synapse_new.prev_delta = synapse_old.prev_delta
  #     synapse_new.prev_delta_w = synapse_old.prev_delta_w
  #     synapse_new.m_current = synapse_old.m_current
  #     synapse_new.v_current = synapse_old.v_current
  #     synapse_new.m_prev = synapse_old.m_prev
  #     synapse_new.v_prev = synapse_old.v_prev
  #     return synapse_new
  #   end

  #   def inspect
  #     pp @weight
  #     pp @source_neuron
  #     pp @dest_neuron
  #   end
  # end
end
