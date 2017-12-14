module SHAInet
  class NeuralNet
    def initialize(input_layer : Int32, hidden_layer : Array(Int32), output_layer : Int32)
      @input_layer = Array(Neuron).new(input_layer, Neuron.new)
      @output_layer = Array(Neuron).new(output_layer, Neuron.new)
      @hidden_layer = Array(Array(Neuron)).new
      hidden_layer.each do |l|
        @hidden_layer << Array(Neuron).new(l, Neuron.new)
      end
    end

    def inspect
      p @input_layer
      p @hidden_layer
      p @output_layer
    end
  end
end
