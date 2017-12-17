module SHAInet
  class Synapse
    property weight : Float64, bias : Float64, :source_neuron, :dest_neuron

    def initialize(@source_neuron : Neuron, @dest_neuron : Neuron)
      @weight = rand(0.0..1.0)
      @bias = rand(-1.0..1.0)
    end

    def randomize_weight
      @weight = rand(0.0..1.0).to_f64
    end

    def randomize_bias
      @bias = rand(-1.0..1.0).to_f64
    end

    def update_weight(value : Float64)
      @weight = value
    end

    def update_bias(value : Float64)
      @bias = value
    end

    def propagate : Array(Float64)
      # Transfer data from source_neuron to dest_neuron using weight & bias values
      new_data = [] of Float64
      puts "inside Propagate"
      @source_neuron.memory.each do |x|
        puts "source mem: #{x}"
        value = x*@weight + @bias
        new_data << value
      end
      puts "After source mem"

      case @source_neuron.n_type
      when :memory
        return new_data
      when :eraser
        new_memory = [] of Float64
        (0..new_data.size - 1).each do |x|
          result = (-1)*new_data[x]
          new_memory << result
        end
        return new_memory
      else
        raise "Other types of neurons are not supported yet!"
      end
    end

    def inspect
      pp @weight
      pp @bias
      pp @source_neuron
      pp @dest_neuron
    end
  end
end
