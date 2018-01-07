module SHAInet
  class SF_layer
    getter output : Array(Array(Array(Float64)))

    def initialize(input_layer : CNN_layer)
      @output = input_layer.output
    end

    def activate
      soft_max(@output)
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
