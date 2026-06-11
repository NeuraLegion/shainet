require "./simple_matrix"
{% if flag?(:enable_cuda) %}
  require "../cuda"
{% else %}
  require "../cuda_stub"
{% end %}

module SHAInet
  # Q8_0-style quantized weight matrix for GPU inference.
  #
  # Logical shape matches the fp32 weight it replaces: [rows = K (in_features),
  # cols = N (out_features)], so it is used the same way as `x[M,K] * w[K,N]`
  # producing `y[M,N]`.
  #
  # Storage on device:
  #   * int8 weights `q` laid out row-major [N, K] (out-major)
  #   * fp32 `scales` laid out [N, ceil(K/BLOCK)] — one scale per BLOCK (=32)
  #     contiguous K elements per output column.
  #
  # Quantization is symmetric per block: scale = max_abs / 127,
  # q = round(v / scale), clamped to [-127, 127].
  class QuantizedCudaMatrix
    BLOCK = 32

    getter rows : Int32   # K (in_features)
    getter cols : Int32   # N (out_features)
    getter blocks : Int32 # ceil(K / BLOCK)
    getter q_ptr : Pointer(Int8)
    getter scale_ptr : Pointer(Float32)

    @q_bytes : UInt64
    @scale_bytes : UInt64

    def initialize(@rows : Int32, @cols : Int32)
      raise RuntimeError.new("QuantizedCudaMatrix requires CUDA to be available") unless CUDA.fully_available?
      @blocks = (@rows + BLOCK - 1) // BLOCK
      @q_bytes = @cols.to_u64 * @rows.to_u64
      @scale_bytes = @cols.to_u64 * @blocks.to_u64 * 4_u64

      qp = Pointer(Int8).null
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

    # Approximate device memory footprint in bytes (int8 weights + fp32 scales).
    def device_bytes : UInt64
      @q_bytes + @scale_bytes
    end

    # Quantize a host fp32 weight matrix [K, N] into this Q8 layout and upload.
    def self.from_simple(w : SimpleMatrix) : QuantizedCudaMatrix
      k = w.rows
      n = w.cols
      qm = new(k, n)
      blocks = qm.blocks

      wdata = w.data # row-major [K, N], element [r, c] at r * n + c
      q_host = Array(Int8).new(n * k, 0_i8)
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

          scale = max_abs > 0.0_f32 ? (max_abs / 127.0_f32) : 1.0_f32
          inv = 1.0_f32 / scale
          s_host[col * blocks + b] = scale

          row_base = col * k
          kk = base
          while kk < lim
            qv = (wdata[kk * n + col] * inv).round
            qv = 127.0_f32 if qv > 127.0_f32
            qv = -127.0_f32 if qv < -127.0_f32 # symmetric range, avoid -128
            q_host[row_base + kk] = qv.to_i8
            kk += 1
          end
        end
      end

      qm.upload(q_host, s_host)
      qm
    end

    # Copy host int8 weights + fp32 scales to device.
    def upload(q_host : Array(Int8), s_host : Array(Float32))
      raise ArgumentError.new("q size mismatch") unless q_host.size.to_u64 == @q_bytes
      raise ArgumentError.new("scale size mismatch") unless (s_host.size.to_u64 * 4_u64) == @scale_bytes
      CUDA.memcpy(@q_ptr.as(Pointer(Void)), q_host.to_unsafe.as(Pointer(Void)), @q_bytes, CUDA::MemcpyKind::HostToDevice)
      CUDA.memcpy(@scale_ptr.as(Pointer(Void)), s_host.to_unsafe.as(Pointer(Void)), @scale_bytes, CUDA::MemcpyKind::HostToDevice)
      self
    end

    # y[M,N] = x[M,K] * dequant(self), computed on GPU. Returns a new CudaMatrix.
    def gemv(x : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("dimension mismatch: x.cols=#{x.cols} vs K=#{@rows}") unless x.cols == @rows
      raise RuntimeError.new("Q8 gemv requires valid device pointers") if @q_ptr.null? || @scale_ptr.null?

      # Ensure activation is resident on device (cheap no-op when already synced).
      x.sync_to_device!("q8_gemv_in") unless x.device_dirty?

      result = CudaMatrix.new(x.rows, @cols)
      CUDA.gemm_q8_f32(x.device_ptr.not_nil!, @q_ptr, @scale_ptr,
        result.device_ptr.not_nil!, x.rows, @cols, @rows)
      result.mark_device_dirty!
      result
    end
  end
end
