module SHAInet
  # BatchProcessor helps run multiple MatrixLayer forward passes
  # in a convenient single call. It simply forwards each input
  # through the corresponding layer and returns the outputs.
  class BatchProcessor
    alias MatrixData = SimpleMatrix | CudaMatrix

    def self.process_batch(layers : Array(MatrixLayer), inputs : Array(MatrixData))
      raise ArgumentError.new("size mismatch") unless layers.size == inputs.size

      outputs = [] of MatrixData
      layers.each_with_index do |layer, idx|
        outputs << layer.forward(inputs[idx])
      end
      outputs
    end
  end
end
