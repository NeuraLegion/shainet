module SHAInet
  class Neuron
    property :matrix

    def initialze
      @matrix = Array(Float64).new
    end

    def update_matrix(wight : Float64, new_matrix : Array(Float64))
      # Update the matrix using wight relevance
      # @matrix = new_matrix * wight
    end
  end
end
