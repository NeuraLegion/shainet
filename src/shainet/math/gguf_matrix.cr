require "./cuda_matrix"           # CudaMatrix used in the GEMV interface
require "./quantized_cuda_matrix" # defines the QuantizedWeight module (included below)
require "../gguf"                 # GGUF::GGMLType
{% if flag?(:enable_cuda) %}
  require "../cuda"
{% else %}
  require "../cuda_stub"
{% end %}

module SHAInet
  # GGUF k-quant weight matrix (Q4_K / Q6_K) for GPU inference.
  #
  # Holds the raw GGUF k-quant bytes on the device untouched and dequantizes
  # inside the GEMV kernel. Logical shape matches the fp32 weight it replaces:
  # GGUF stores tensors as [in_features, out_features] row-major, so
  #   rows = K (in_features), cols = N (out_features)
  # and it is used as `y[M,N] = x[M,K] * W[N,K]`.
  class GGUFMatrix
    include QuantizedWeight

    getter rows : Int32               # K (in_features)
    getter cols : Int32               # N (out_features)
    getter ggml_type : GGUF::GGMLType # Q4_K or Q6_K
    getter dev_ptr : Pointer(UInt8)   # device pointer to raw k-quant bytes

    @byte_size : UInt64

    # Allocate device memory, copy the raw k-quant bytes from host, and keep the
    # device pointer. `host_data` points at `byte_size` bytes of GGUF k-quant
    # data laid out as N rows of blocks, each row covering K values.
    def initialize(@rows : Int32, @cols : Int32, @ggml_type : GGUF::GGMLType,
                   host_data : Pointer(UInt8), byte_size)
      @byte_size = byte_size.to_u64
      unless @ggml_type.q4_k? || @ggml_type.q6_k?
        raise ArgumentError.new("GGUFMatrix supports only Q4_K and Q6_K, got #{@ggml_type}")
      end

      dp = Pointer(UInt8).null
      CUDA.malloc(pointerof(dp).as(Pointer(Pointer(Void))), @byte_size)
      @dev_ptr = dp
      CUDA.memcpy(@dev_ptr.as(Pointer(Void)), host_data.as(Pointer(Void)),
        @byte_size, CUDA::MemcpyKind::HostToDevice)
    end

    def finalize
      free!
    end

    # y[M,N] = x[M,K] * dequant(W[N,K]) into a caller-provided result buffer.
    # result must be [x.rows, cols].
    def gemv_into(x : CudaMatrix, result : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("dimension mismatch: x.cols=#{x.cols} vs K=#{@rows}") unless x.cols == @rows
      raise ArgumentError.new("result shape mismatch") unless result.rows == x.rows && result.cols == @cols
      raise RuntimeError.new("GGUF gemv requires a valid device pointer") if @dev_ptr.null?

      # Ensure activation is resident on device (cheap no-op when already synced).
      x.sync_to_device!("gguf_gemv_in") unless x.device_dirty?

      case @ggml_type
      when .q4_k?
        CUDA.gemv_q4k(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .q6_k?
        CUDA.gemv_q6k(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      else
        raise ArgumentError.new("unsupported GGUF type for gemv: #{@ggml_type}")
      end

      result.mark_device_dirty!
      result
    end

    # Same as gemv_into but allocates the result CudaMatrix [x.rows, cols].
    def gemv(x : CudaMatrix) : CudaMatrix
      result = CudaMatrix.new(x.rows, @cols)
      gemv_into(x, result)
    end

    # Free the device allocation if not already freed.
    def free!
      unless @dev_ptr.null?
        CUDA.free(@dev_ptr.as(Pointer(Void)))
        @dev_ptr = Pointer(UInt8).null
      end
    end

    # Device memory footprint in bytes.
    def device_bytes : UInt64
      @byte_size
    end
  end
end
