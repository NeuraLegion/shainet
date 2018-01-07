require "logger"

module SHAInet
  class FC_layer
    property :neurons, :all_synapses
    getter :output

    def initialize(l_size : Int32, input_layer : CNN_layer, @activation_function : Proc(GenNum, Array(Float64)) = SHAInet.sigmoid, @logger : Logger = Logger.new(STDOUT))
      @neurons = Array(Neuron).new(l_size) { Neuron.new("memory") }
      @output = Array(Float64).new(l_size) { 0.0 }
      @all_synapses = Array(Synapse).new

      # Connect the last layer to the output layer (fully connect)
      @neurons.each do |neuron2|              # Target neuron
        input_layer.neurons.each do |neuron1| # Source neuron
          synapse = Synapse.new(neuron1, neuron2)
          neuron1.synapses_out << synapse
          neuron2.synapses_in << synapse
          @all_synapses << synapse
        end
      end
    end

    def activate
      @neurons.each_with_index do |neuron, i|
        neuron.activate(@activation_function)
        @output[i] = neuron.activation
      end
    end

    def inspect(what : String)
      case what
      when "weights"
        puts "input layer has no wights"
      when "bias"
      when "neurons"
      end
    end
  end
end
