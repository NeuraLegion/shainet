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
      unless GGUFMatrix.device_type_supported?(@ggml_type)
        raise ArgumentError.new("GGUFMatrix: no device kernel for #{@ggml_type}")
      end

      dp = Pointer(UInt8).null
      CUDA.malloc(pointerof(dp).as(Pointer(Pointer(Void))), @byte_size)
      @dev_ptr = dp
      CUDA.memcpy(@dev_ptr.as(Pointer(Void)), host_data.as(Pointer(Void)),
        @byte_size, CUDA::MemcpyKind::HostToDevice)
    end

    # Create a GGUFMatrix that points into a pre-allocated device buffer.
    # No CUDA.malloc or memcpy -- the data is already on the device in the bulk pool.
    # The pool owns the memory; this matrix must NOT free it.
    def self.from_pool(rows : Int32, cols : Int32, ggml_type : GGUF::GGMLType,
                       dev_ptr : Pointer(UInt8), byte_size : UInt64) : GGUFMatrix
      m = GGUFMatrix.allocate
      m.init_from_pool(rows, cols, ggml_type, dev_ptr, byte_size)
      m
    end

    protected def init_from_pool(@rows : Int32, @cols : Int32, @ggml_type : GGUF::GGMLType,
                                 @dev_ptr : Pointer(UInt8), @byte_size : UInt64)
      @pool_owned = true
    end

    @pool_owned : Bool = false

    # Debug tracing flag, read ONCE at class init.
    #
    # This was ENV["SHAINET_DEBUG"]? evaluated inline on every gemv_into call, TWICE. A Crystal ENV
    # lookup calls getenv -- a linear scan of the environment -- and allocates a String for the
    # result. A generated token issues on the order of five hundred GEMV calls across the resident
    # layers, so the untaken branch of a debug print cost ~1000 getenv calls and ~1000 short-lived
    # allocations per token, for nothing.
    @@debug : Bool = ENV["SHAINET_DEBUG"]? == "1"

    # Allocate device memory only (no upload). Used by GGUFHostMatrix for its
    # shared scratch buffer -- the data is uploaded per-GEMV via cudaMemcpy.
    def self.new_empty(rows : Int32, cols : Int32, ggml_type : GGUF::GGMLType, byte_size : UInt64) : GGUFMatrix
      m = GGUFMatrix.allocate
      m.initialize_empty(rows, cols, ggml_type, byte_size)
      m
    end

    protected def initialize_empty(@rows : Int32, @cols : Int32, @ggml_type : GGUF::GGMLType, byte_size : UInt64)
      @byte_size = byte_size
      unless GGUFMatrix.device_type_supported?(@ggml_type)
        raise ArgumentError.new("GGUFMatrix: no device kernel for #{@ggml_type}")
      end
      dp = Pointer(UInt8).null
      CUDA.malloc(pointerof(dp).as(Pointer(Pointer(Void))), @byte_size)
      @dev_ptr = dp
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

      if @@debug
        STDERR.puts "  [gguf gemv] #{@ggml_type} M=#{x.rows} N=#{@cols} K=#{@rows} bytes=#{@byte_size} pool=#{@pool_owned}"
        STDERR.flush
      end

      # Ensure activation is resident on device (cheap no-op when already synced).
      x.sync_to_device!("gguf_gemv_in") unless x.device_dirty?

      # M == 1 (decode) is already optimal as a GEMV: the weight is streamed exactly once. For
      # M > 1 the GEMV kernel gives each (output, token) pair its own block, so it re-dequantizes
      # the whole weight per token and reuses nothing -- measured flat at 0.54 ms/token from M=1 to
      # M=256 on a 47.8 MB Q4_K weight, i.e. 54 GB of reads for a 1200-token prefill of ONE matmul.
      # Dequantizing once into a scratch tile and letting cuBLAS do the GEMM reads the weight once.
      if x.rows > 1 && gemm_capable?
        return gemm_into(x, result)
      end

      case @ggml_type
      when .iq4_xs?
        CUDA.gemv_iq4xs(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq3_s?
        CUDA.gemv_iq3s(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq3_xxs?
        CUDA.gemv_iq3xxs(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .q2_k?
        CUDA.gemv_q2k_lb(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq2_xxs?
        CUDA.gemv_iq2xxs(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq2_xs?
        CUDA.gemv_iq2xs(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq2_s?
        CUDA.gemv_iq2s(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq1_m?
        CUDA.gemv_iq1m(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq1_s?
        CUDA.gemv_iq1s(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
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

    # Scratch for the dequantized weight tile, shared across every GGUFMatrix.
    #
    # One buffer rather than one per weight: prefill runs the matmuls sequentially, so a single
    # tile is live at a time, and a per-weight buffer would cost hundreds of MB of VRAM on a card
    # that is already nearly full with the model.
    @@scratch : Pointer(Float32) = Pointer(Float32).null
    @@scratch_floats : Int32 = 0
    @@gemm_supported : Bool? = nil

    # Rows dequantized per cuBLAS call. 1024 x K=5120 fp32 is 20 MB, which is small next to the
    # model yet large enough that the GEMM is not launch-bound.
    SCRATCH_ROWS = 1024

    def self.release_scratch!
      unless @@scratch.null?
        CUDA.free(@@scratch.as(Pointer(Void)))
        @@scratch = Pointer(Float32).null
        @@scratch_floats = 0
      end
    end

    # Which quantization types have a device GEMV.
    #
    # Kept as one predicate because three places need the same answer: both constructors, and
    # stage_host when it decides whether a host weight can be borrowed onto the card. A type missing
    # here must be transcoded at load rather than reaching a kernel that cannot read it.
    def self.device_type_supported?(t : GGUF::GGMLType) : Bool
      return t.q4_k? || t.q6_k? if @@force_transcode_iq
      t.q4_k? || t.q6_k? || t.iq4_xs? || t.iq3_s? || t.iq3_xxs? ||
        t.q2_k? || t.iq2_xxs? || t.iq2_xs? || t.iq2_s? || t.iq1_m? || t.iq1_s?
    end

    # Diagnostic switch: treat the i-quant types as unsupported so every one of them is transcoded
    # through the scalar reference instead of reaching a kernel. The references are independently
    # verified against a differently-quantized build of the same model, so this separates "a kernel is
    # wrong" from "the loader wires something wrong" -- the two explanations that survive when every
    # component passes its own test but the model still does not predict.
    @@force_transcode_iq : Bool = ENV.fetch("SHAINET_IQ_FORCE_TRANSCODE", "0") == "1"

    # ── One reusable device staging slot for weights that live on the HOST ──
    #
    # A host-resident layer is the right choice when DECODING: at one row it streams its weights at
    # ~69 GB/s of DDR5 and the arithmetic is memory-bound, so the CPU keeps up. PREFILL is a
    # different problem. At 256 rows the same projection is COMPUTE-bound -- ffn_gate plus ffn_up is
    # 46 GFLOP per chunk -- and the CPU is about two orders of magnitude off the GPU there. Measured
    # on a 301-token prefill: the 24 host-layer ffn_gate_up calls took 5520 ms of the 5590 ms spent
    # in that phase, while the 104 device-layer calls took 73 ms between them.
    #
    # Above a batch threshold it is therefore cheaper to SEND the quantized weight to the card and
    # use the dequant+cuBLAS path, because the upload amortizes over the rows: ffn_gate is 47.8 MB,
    # roughly 4 ms over PCIe, against ~230 ms of CPU GEMM. One slot suffices because a staged weight
    # is consumed before the next is staged, and the slot is grown rather than reallocated per call.
    @@stage_ptr = Pointer(UInt8).null
    @@stage_bytes = 0_u64

    # Upload host-resident k-quant bytes into the staging slot and return a NON-OWNING view of them so
    # the caller can use the device GEMM path. Returns nil when the weight cannot be staged, which
    # leaves the host path as the fallback rather than turning it into a failure.
    def self.stage_host(rows : Int32, cols : Int32, ggml_type : GGUF::GGMLType,
                        host_data : Pointer(UInt8), byte_size : UInt64) : GGUFMatrix?
      return unless CUDA.fully_available?
      return unless GGUFMatrix.device_type_supported?(ggml_type)
      if @@stage_bytes < byte_size
        unless @@stage_ptr.null?
          CUDA.free(@@stage_ptr.as(Pointer(Void)))
          @@stage_ptr = Pointer(UInt8).null
          @@stage_bytes = 0_u64
        end
        p = Pointer(Void).null
        CUDA.malloc(pointerof(p), byte_size)
        return if p.null?
        @@stage_ptr = p.as(Pointer(UInt8))
        @@stage_bytes = byte_size
      end
      CUDA.memcpy(@@stage_ptr.as(Pointer(Void)), host_data.as(Pointer(Void)),
        byte_size, CUDA::MemcpyKind::HostToDevice)
      view = allocate
      view.init_view(rows, cols, ggml_type, @@stage_ptr, byte_size)
      view
    end

    # Adopt an existing device pointer. @pool_owned suppresses the free, so the staging slot outlives
    # every view taken of it.
    protected def init_view(@rows : Int32, @cols : Int32, @ggml_type : GGUF::GGMLType,
                            @dev_ptr : Pointer(UInt8), @byte_size : UInt64)
      @pool_owned = true
    end

    protected def self.scratch(floats : Int32) : Pointer(Float32)
      if @@scratch_floats < floats
        release_scratch!
        p = Pointer(Void).null
        CUDA.malloc(pointerof(p), (floats.to_u64 * 4))
        @@scratch = p.as(Pointer(Float32))
        @@scratch_floats = floats
      end
      @@scratch
    end

    private def gemm_capable? : Bool
      sup = @@gemm_supported
      return sup unless sup.nil?
      @@gemm_supported = sup = CUDA.fully_available? && CUDA.dequant_k_rows_available?
      sup
    end

    # Batched matmul for M > 1: dequantize the weight in row chunks and let cuBLAS multiply.
    #
    # result[M, N] = x[M, K] * dequant(W[N, K])^T. cuBLAS is column-major, so the row-major
    # result [M, N] is addressed as a column-major [N, M]: with the dequantized chunk as the
    # transposed operand and ldc set to the FULL N, each chunk writes straight into its own
    # columns of the finished result.
    def gemm_into(x : CudaMatrix, result : CudaMatrix) : CudaMatrix
      k = @rows
      chunk = SCRATCH_ROWS > @cols ? @cols : SCRATCH_ROWS
      buf = GGUFMatrix.scratch(chunk * k)
      handle = CUDA.create_handle
      begin
        xp = x.device_ptr.not_nil!
        rp = result.device_ptr.not_nil!
        row0 = 0
        while row0 < @cols
          rows = @cols - row0
          rows = chunk if rows > chunk
          case @ggml_type
          when .iq4_xs?  then CUDA.dequant_iq4xs_rows(@dev_ptr, buf, row0, rows, k)
          when .iq3_s?   then CUDA.dequant_iq3s_rows(@dev_ptr, buf, row0, rows, k)
          when .iq3_xxs? then CUDA.dequant_iq3xxs_rows(@dev_ptr, buf, row0, rows, k)
          when .q2_k?    then CUDA.dequant_q2k_lb_rows(@dev_ptr, buf, row0, rows, k)
          when .iq2_xxs? then CUDA.dequant_iq2xxs_rows(@dev_ptr, buf, row0, rows, k)
          when .iq2_xs?  then CUDA.dequant_iq2xs_rows(@dev_ptr, buf, row0, rows, k)
          when .iq2_s?   then CUDA.dequant_iq2s_rows(@dev_ptr, buf, row0, rows, k)
          when .iq1_m?   then CUDA.dequant_iq1m_rows(@dev_ptr, buf, row0, rows, k)
          when .iq1_s?   then CUDA.dequant_iq1s_rows(@dev_ptr, buf, row0, rows, k)
          when .q4_k?    then CUDA.dequant_q4k_rows(@dev_ptr, buf, row0, rows, k)
          when .q6_k?    then CUDA.dequant_q6k_rows(@dev_ptr, buf, row0, rows, k)
          else                raise ArgumentError.new("unsupported GGUF type for gemm: #{@ggml_type}")
          end
          CUDA.gemm_tn(handle, buf, xp, rp + row0,
            rows, x.rows, k, k, k, @cols)
          row0 += rows
        end
      ensure
        CUDA.destroy_handle(handle)
      end
      result.mark_device_dirty!
      result
    end

    # Free the device allocation if not already freed.
    def free!
      unless @dev_ptr.null? || @pool_owned
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
