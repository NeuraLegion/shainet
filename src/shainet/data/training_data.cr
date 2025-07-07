{% if flag?(:enable_cuda) %}
  require "../cuda"
{% else %}
  require "../cuda_stub"
{% end %}
require "../math/cuda_matrix"
require "../math/gpu_memory"

module SHAInet
  # TrainingData represents normalized datasets used for standard
  # in-memory training. When `preload_gpu` is enabled and CUDA is
  # available, the normalized inputs and outputs can be converted to
  # `CudaMatrix` instances once up front to avoid per-sample CPU->GPU
  # transfers during training.
  class TrainingData < Data
    property? preload_gpu
    @gpu_inputs : Array(CudaMatrix) = [] of CudaMatrix
    @gpu_outputs : Array(CudaMatrix) = [] of CudaMatrix

    def initialize(@inputs : Array(Array(Float64)), @outputs : Array(Array(Float64)), @preload_gpu : Bool = false)
      super(@inputs, @outputs)
    end

    # Convert all normalized data to CudaMatrix and store it. This
    # should be called after the data has been normalized.
    def preload_gpu!
      return unless CUDA.fully_available?
      return if @gpu_inputs.size == @normalized_inputs.size && @gpu_outputs.size == @normalized_outputs.size

      @gpu_inputs = Array(CudaMatrix).new(@normalized_inputs.size) do |idx|
        row = @normalized_inputs[idx]
        mat = CudaMatrix.new(1, row.size)
        GPUMemory.to_gpu!(row, mat)
        mat
      end

      @gpu_outputs = Array(CudaMatrix).new(@normalized_outputs.size) do |idx|
        row = @normalized_outputs[idx]
        mat = CudaMatrix.new(1, row.size)
        GPUMemory.to_gpu!(row, mat)
        mat
      end
      @preload_gpu = true
    end

    # Return training pairs either as arrays of Float64 or as GPU
    # matrices when preloaded.
    def data
      if @preload_gpu && CUDA.fully_available?
        arr = [] of Array(CudaMatrix)
        @gpu_inputs.each_with_index do |input, i|
          arr << [input, @gpu_outputs[i]]
        end
        arr
      else
        arr = [] of Array(Array(Float64))
        @normalized_inputs.each_with_index do |_, i|
          arr << [@normalized_inputs[i], @normalized_outputs[i]]
        end
        arr
      end
    end
  end
end
