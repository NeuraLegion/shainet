require "./simple_matrix"
require "./quantized_cuda_matrix" # defines QuantizedWeight (included below)
{% if flag?(:enable_cuda) %}
  require "../cuda"
{% else %}
  require "../cuda_stub"
{% end %}

module SHAInet
  # Q4_0-style 4-bit quantized weight matrix for GPU inference.
  #
  # Logical shape matches the fp32 weight it replaces: [rows = K (in_features),
  # cols = N (out_features)], used as `x[M,K] * w[K,N]` producing `y[M,N]`.
  #
  # Storage on device:
  #   * 4-bit weights packed two-per-byte along K, laid out [N, ceil(K/2)]
  #     (out-major). Byte (k/2) for column n holds k in the low nibble and k+1
  #     in the high nibble, each stored as (value + 8) in 0..15.
  #   * fp32 `scales` laid out [N, ceil(K/BLOCK)] — one scale per BLOCK (=32)
  #     contiguous K elements per output column.
  #
  # Quantization is symmetric per block: scale = max_abs / 7,
  # value = round(v / scale) clamped to [-7, 7], stored nibble = value + 8.
  class Q4CudaMatrix
    include QuantizedWeight
    BLOCK = 32

    getter rows : Int32   # K (in_features)
    getter cols : Int32   # N (out_features)
    getter blocks : Int32 # ceil(K / BLOCK)
    getter kbytes : Int32 # ceil(K / 2) packed bytes per output column
    getter q_ptr : Pointer(UInt8)
    getter scale_ptr : Pointer(Float32)

    @q_bytes : UInt64
    @scale_bytes : UInt64

    def initialize(@rows : Int32, @cols : Int32)
      raise RuntimeError.new("Q4CudaMatrix requires CUDA to be available") unless CUDA.fully_available?
      @blocks = (@rows + BLOCK - 1) // BLOCK
      @kbytes = (@rows + 1) // 2
      @q_bytes = @cols.to_u64 * @kbytes.to_u64
      @scale_bytes = @cols.to_u64 * @blocks.to_u64 * 4_u64

      qp = Pointer(UInt8).null
      CUDA.malloc(pointerof(qp).as(Pointer(Pointer(Void))), @q_bytes)
      @q_ptr = qp

      sp = Pointer(Float32).null
      CUDA.malloc(pointerof(sp).as(Pointer(Pointer(Void))), @scale_bytes)
      @scale_ptr = sp
    end

    def finalize
      CUDA.free(@q_ptr.as(Pointer(Void))) unless @q_ptr.null?
      CUDA.free(@scale_ptr.as(Pointer(Void))) unless @scale_ptr.null?
    end

    # Approximate device memory footprint in bytes (4-bit weights + fp32 scales).
    def device_bytes : UInt64
      @q_bytes + @scale_bytes
    end

    # Quantize a host fp32 weight matrix [K, N] into this Q4 layout and upload.
    def self.from_simple(w : SimpleMatrix) : Q4CudaMatrix
      k = w.rows
      n = w.cols
      qm = new(k, n)
      blocks = qm.blocks
      kbytes = qm.kbytes

      wdata = w.data # row-major [K, N], element [r, c] at r * n + c
      q_host = Array(UInt8).new(n * kbytes, 0_u8)
      s_host = Array(Float32).new(n * blocks, 0.0_f32)

      n.times do |col|
        blocks.times do |b|
          base = b * BLOCK
          lim = Math.min(base + BLOCK, k)

          max_abs = 0.0_f32
          kk = base
          while kk < lim
            v = wdata[kk * n + col].abs
            max_abs = v if v > max_abs
            kk += 1
          end

          scale = max_abs > 0.0_f32 ? (max_abs / 7.0_f32) : 1.0_f32
          inv = 1.0_f32 / scale
          s_host[col * blocks + b] = scale

          row_base = col * kbytes
          kk = base
          while kk < lim
            qv = (wdata[kk * n + col] * inv).round
            qv = 7.0_f32 if qv > 7.0_f32
            qv = -7.0_f32 if qv < -7.0_f32 # symmetric range
            nib = (qv.to_i + 8) & 0x0F      # store value+8 in 0..15
            byte_idx = row_base + (kk >> 1)
            if (kk & 1) == 0
              q_host[byte_idx] = (q_host[byte_idx] & 0xF0_u8) | nib.to_u8
            else
              q_host[byte_idx] = (q_host[byte_idx] & 0x0F_u8) | (nib.to_u8 << 4)
            end
            kk += 1
          end
        end
      end

      qm.upload(q_host, s_host)
      qm
    end

    # Copy host 4-bit weights + fp32 scales to device.
    def upload(q_host : Array(UInt8), s_host : Array(Float32))
      raise ArgumentError.new("q size mismatch") unless q_host.size.to_u64 == @q_bytes
      raise ArgumentError.new("scale size mismatch") unless (s_host.size.to_u64 * 4_u64) == @scale_bytes
      CUDA.memcpy(@q_ptr.as(Pointer(Void)), q_host.to_unsafe.as(Pointer(Void)), @q_bytes, CUDA::MemcpyKind::HostToDevice)
      CUDA.memcpy(@scale_ptr.as(Pointer(Void)), s_host.to_unsafe.as(Pointer(Void)), @scale_bytes, CUDA::MemcpyKind::HostToDevice)
      self
    end

    # y[M,N] = x[M,K] * dequant(self), computed on GPU. Returns a new CudaMatrix.
    def gemv(x : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("dimension mismatch: x.cols=#{x.cols} vs K=#{@rows}") unless x.cols == @rows
      raise RuntimeError.new("Q4 gemv requires valid device pointers") if @q_ptr.null? || @scale_ptr.null?

      x.sync_to_device!("q4_gemv_in") unless x.device_dirty?

      result = CudaMatrix.new(x.rows, @cols)
      CUDA.gemm_q4_f32(x.device_ptr.not_nil!, @q_ptr, @scale_ptr,
        result.device_ptr.not_nil!, x.rows, @cols, @rows)
      result.mark_device_dirty!
      result
    end

    # Same as gemv but writes into a caller-provided result buffer, avoiding a
    # per-call device allocation. result must be [x.rows, cols].
    def gemv_into(x : CudaMatrix, result : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("dimension mismatch: x.cols=#{x.cols} vs K=#{@rows}") unless x.cols == @rows
      raise ArgumentError.new("result shape mismatch") unless result.rows == x.rows && result.cols == @cols
      raise RuntimeError.new("Q4 gemv requires valid device pointers") if @q_ptr.null? || @scale_ptr.null?

      x.sync_to_device!("q4_gemv_in") unless x.device_dirty?
      CUDA.gemm_q4_f32(x.device_ptr.not_nil!, @q_ptr, @scale_ptr,
        result.device_ptr.not_nil!, x.rows, @cols, @rows)
      result.mark_device_dirty!
      result
    end
  end
end
