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
    getter supers : Int32 # ceil(blocks / SUPER)
    getter kbytes : Int32 # ceil(K / 2) packed bytes per output column
    getter q_ptr : Pointer(UInt8)
    getter d_ptr : Pointer(Float32) # one fp32 scale per super-block
    getter sub_ptr : Pointer(UInt8) # one byte per block, relative to its super-block

    @q_bytes : UInt64
    @d_bytes : UInt64
    @sub_bytes : UInt64

    def initialize(@rows : Int32, @cols : Int32)
      raise RuntimeError.new("Q4CudaMatrix requires CUDA to be available") unless CUDA.fully_available?
      @blocks = (@rows + BLOCK - 1) // BLOCK
      @supers = (@blocks + SUPER - 1) // SUPER
      @kbytes = (@rows + 1) // 2
      @q_bytes = @cols.to_u64 * @kbytes.to_u64
      @d_bytes = @cols.to_u64 * @supers.to_u64 * 4_u64
      @sub_bytes = @cols.to_u64 * @blocks.to_u64

      qp = Pointer(UInt8).null
      CUDA.malloc(pointerof(qp).as(Pointer(Pointer(Void))), @q_bytes)
      @q_ptr = qp

      dp = Pointer(Float32).null
      CUDA.malloc(pointerof(dp).as(Pointer(Pointer(Void))), @d_bytes)
      @d_ptr = dp

      sp = Pointer(UInt8).null
      CUDA.malloc(pointerof(sp).as(Pointer(Pointer(Void))), @sub_bytes)
      @sub_ptr = sp
    end

    def finalize
      free!
    end

    # Explicitly free device memory now (used by the expert cache on eviction so
    # VRAM is reclaimed immediately rather than waiting for GC). Idempotent.
    def free!
      CUDA.free(@q_ptr.as(Pointer(Void))) unless @q_ptr.null?
      CUDA.free(@d_ptr.as(Pointer(Void))) unless @d_ptr.null?
      CUDA.free(@sub_ptr.as(Pointer(Void))) unless @sub_ptr.null?
      @q_ptr = Pointer(UInt8).null
      @d_ptr = Pointer(Float32).null
      @sub_ptr = Pointer(UInt8).null
    end

    # Device bytes a Q4 weight of shape [k, n] would occupy: 4-bit weights, one
    # fp32 scale per super-block, one byte per block. Without allocating — used
    # for cache budget accounting.
    def self.device_bytes_for(k : Int32, n : Int32) : UInt64
      blocks = (k + BLOCK - 1) // BLOCK
      supers = (blocks + SUPER - 1) // SUPER
      kbytes = (k + 1) // 2
      n.to_u64 * kbytes.to_u64 + n.to_u64 * supers.to_u64 * 4_u64 + n.to_u64 * blocks.to_u64
    end

    # Approximate device memory footprint in bytes.
    def device_bytes : UInt64
      @q_bytes + @d_bytes + @sub_bytes
    end

    # Number of 32-weight blocks that share one super-block scale. The per-block
    # scale is then stored as a single byte relative to that super-block scale
    # instead of a full fp32, which is where the size saving comes from:
    #
    #   per 256 weights, old: 128 B nibbles + 8 * 4 B fp32 scales = 160 B = 5.000 bpw
    #   per 256 weights, new: 128 B nibbles + 4 B fp32 d + 8 * 1 B = 140 B = 4.375 bpw
    #
    # 12.5% smaller. A super-block scale in fp16 would reach 4.3125 bpw, but the
    # extra 1.4% is not worth hand-rolling fp16 conversion here and __half2float
    # in both kernels.
    SUPER = 8

    # Quantize a host fp32 weight matrix [K, N] into the super-block Q4 layout.
    # Returns the packed nibbles, the per-super-block fp32 scales, and the
    # per-block byte sub-scales. The effective scale of block b is
    #
    #   d[b / SUPER] * sub[b] / 255
    #
    # Weights are quantized against that EFFECTIVE scale, not against the ideal
    # per-block scale: quantizing against the ideal and then rounding the scale
    # would apply the scale error twice.
    def self.pack(w : SimpleMatrix) : Tuple(Array(UInt8), Array(Float32), Array(UInt8))
      k = w.rows
      n = w.cols
      blocks = (k + BLOCK - 1) // BLOCK
      supers = (blocks + SUPER - 1) // SUPER
      kbytes = (k + 1) // 2

      wdata = w.data # row-major [K, N], element [r, c] at r * n + c
      q_host = Array(UInt8).new(n * kbytes, 0_u8)
      d_host = Array(Float32).new(n * supers, 0.0_f32)
      sub_host = Array(UInt8).new(n * blocks, 0_u8)

      # Scratch reused across columns so packing a large weight does not churn.
      ideal = Array(Float32).new(SUPER, 0.0_f32)

      n.times do |col|
        supers.times do |s|
          b0 = s * SUPER
          b1 = Math.min(b0 + SUPER, blocks)

          # Pass 1: the ideal per-block scale, and the super-block maximum.
          dmax = 0.0_f32
          (b0...b1).each_with_index do |b, i|
            base = b * BLOCK
            lim = Math.min(base + BLOCK, k)
            max_abs = 0.0_f32
            kk = base
            while kk < lim
              v = wdata[kk * n + col].abs.to_f32
              max_abs = v if v > max_abs
              kk += 1
            end
            sc = max_abs > 0.0_f32 ? (max_abs / 7.0_f32).to_f32 : 0.0_f32
            ideal[i] = sc
            dmax = sc if sc > dmax
          end

          # A super-block of all zeros: keep d positive so the kernel's multiply is
          # well defined, and leave every sub-scale at its minimum.
          d = dmax > 0.0_f32 ? dmax : 1.0_f32
          d_host[col * supers + s] = d
          inv_d = 255.0_f32 / d

          # Pass 2: byte sub-scale, then quantize against the effective scale.
          (b0...b1).each_with_index do |b, i|
            byte = (ideal[i] * inv_d).round.to_i
            byte = 1 if byte < 1 # 0 would make the block unrepresentable
            byte = 255 if byte > 255
            sub_host[col * blocks + b] = byte.to_u8

            eff = d * byte.to_f32 / 255.0_f32
            inv = 1.0_f32 / eff
            base = b * BLOCK
            lim = Math.min(base + BLOCK, k)
            row_base = col * kbytes
            kk = base
            while kk < lim
              qv = (wdata[kk * n + col].to_f32 * inv).round
              qv = 7.0_f32 if qv > 7.0_f32
              qv = -7.0_f32 if qv < -7.0_f32 # symmetric range
              nib = (qv.to_i + 8) & 0x0F     # store value+8 in 0..15
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
      end

      {q_host, d_host, sub_host}
    end

    # Quantize a host fp32 weight matrix [K, N] into this Q4 layout and upload.
    def self.from_simple(w : SimpleMatrix) : Q4CudaMatrix
      qm = new(w.rows, w.cols)
      q_host, d_host, sub_host = pack(w)
      qm.upload(q_host, d_host, sub_host)
      qm
    end

    # Copy host 4-bit weights, super-block scales and byte sub-scales to device.
    def upload(q_host : Array(UInt8), d_host : Array(Float32), sub_host : Array(UInt8))
      raise ArgumentError.new("q size mismatch") unless q_host.size.to_u64 == @q_bytes
      raise ArgumentError.new("d size mismatch") unless (d_host.size.to_u64 * 4_u64) == @d_bytes
      raise ArgumentError.new("sub size mismatch") unless sub_host.size.to_u64 == @sub_bytes
      CUDA.memcpy(@q_ptr.as(Pointer(Void)), q_host.to_unsafe.as(Pointer(Void)), @q_bytes, CUDA::MemcpyKind::HostToDevice)
      CUDA.memcpy(@d_ptr.as(Pointer(Void)), d_host.to_unsafe.as(Pointer(Void)), @d_bytes, CUDA::MemcpyKind::HostToDevice)
      CUDA.memcpy(@sub_ptr.as(Pointer(Void)), sub_host.to_unsafe.as(Pointer(Void)), @sub_bytes, CUDA::MemcpyKind::HostToDevice)
      self
    end

    # y[M,N] = x[M,K] * dequant(self), computed on GPU. Returns a new CudaMatrix.
    def gemv(x : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("dimension mismatch: x.cols=#{x.cols} vs K=#{@rows}") unless x.cols == @rows
      raise RuntimeError.new("Q4 gemv requires valid device pointers") if @q_ptr.null? || @d_ptr.null? || @sub_ptr.null?

      x.sync_to_device!("q4_gemv_in") unless x.device_dirty?

      result = CudaMatrix.new(x.rows, @cols)
      CUDA.gemm_q4_f32(x.device_ptr.not_nil!, @q_ptr, @d_ptr, @sub_ptr,
        result.device_ptr.not_nil!, x.rows, @cols, @rows)
      result.mark_device_dirty!
      result
    end

    # Same as gemv but writes into a caller-provided result buffer, avoiding a
    # per-call device allocation. result must be [x.rows, cols].
    def gemv_into(x : CudaMatrix, result : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("dimension mismatch: x.cols=#{x.cols} vs K=#{@rows}") unless x.cols == @rows
      raise ArgumentError.new("result shape mismatch") unless result.rows == x.rows && result.cols == @cols
      raise RuntimeError.new("Q4 gemv requires valid device pointers") if @q_ptr.null? || @d_ptr.null? || @sub_ptr.null?

      x.sync_to_device!("q4_gemv_in") unless x.device_dirty?
      CUDA.gemm_q4_f32(x.device_ptr.not_nil!, @q_ptr, @d_ptr, @sub_ptr,
        result.device_ptr.not_nil!, x.rows, @cols, @rows)
      result.mark_device_dirty!
      result
    end
  end
end
