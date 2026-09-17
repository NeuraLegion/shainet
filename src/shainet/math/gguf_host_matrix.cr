module SHAInet
  # Host-resident GGUF k-quant weight matrix backed by an mmap'd file region.
  # Zero-copy on the host side: no allocation, no dequant. For each GEMV the raw
  # Q4_K/Q6_K bytes are streamed to a device scratch buffer via cudaMemcpy, the
  # GEMV runs on the GPU, and the scratch is reused for the next weight.
  #
  # This is the CPU-layer equivalent of GGUFMatrix (device-resident). Together
  # they implement the layer-level GPU/CPU split: GPU layers use GGUFMatrix,
  # CPU layers use GGUFHostMatrix.
  class GGUFHostMatrix
    include QuantizedWeight

    getter rows : Int32
    getter cols : Int32
    getter ggml_type : GGUF::GGMLType

    @host_ptr : Pointer(UInt8)
    @byte_size : UInt64

    def initialize(@rows, @cols, @ggml_type, @host_ptr, @byte_size)
      unless @ggml_type == GGUF::GGMLType::Q4_K || @ggml_type == GGUF::GGMLType::Q6_K
        raise ArgumentError.new("GGUFHostMatrix: unsupported type #{@ggml_type}")
      end
    end

    # Shared device scratch buffer, keyed by (rows, cols, type) to avoid
    # reallocating on every GEMV.
    @@scratch = Hash(Tuple(Int32, Int32, GGUF::GGMLType), GGUFMatrix).new
    @@gpu_mutex = Mutex.new

    def device_bytes : UInt64
      0_u64 # host-resident, no permanent device footprint
    end

    def gemv(x : CudaMatrix) : CudaMatrix
      result = CudaMatrix.new(x.rows, @cols)
      gemv_into(x, result)
    end

    def gemv_into(x : CudaMatrix, result : CudaMatrix) : CudaMatrix
      @@gpu_mutex.synchronize do
        # Get or create a device scratch buffer for this weight shape
        key = {@rows, @cols, @ggml_type}
        scratch = @@scratch[key]? || begin
          s = GGUFMatrix.new_empty(@rows, @cols, @ggml_type, @byte_size)
          @@scratch[key] = s
          s
        end
        # Upload the mmap'd data to the scratch buffer
        CUDA.memcpy(scratch.dev_ptr.as(Pointer(Void)), @host_ptr.as(Pointer(Void)),
          @byte_size, CUDA::MemcpyKind::HostToDevice)
        # Run the GEMV on the scratch
        scratch.gemv_into(x, result)
      end
    end

    def free!
      # Nothing to free -- the host pointer is in the mmap'd file
    end

    # Clear all shared scratch buffers (called on model unload)
    def self.clear_scratch!
      @@gpu_mutex.synchronize do
        @@scratch.each_value(&.free!)
        @@scratch.clear
      end
    end
  end
end
