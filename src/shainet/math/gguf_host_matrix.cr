module SHAInet
  # Host-resident GGUF k-quant weight matrix backed by an mmap'd file region.
  # Performs GEMV entirely on the CPU -- zero PCIe transfer, zero GPU involvement.
  #
  # Primary path: AVX2 C kernels (libshainet_cpu_kernels.so) with OpenMP
  # parallelism over output columns. Fallback: Crystal scalar loops.
  class GGUFHostMatrix
    include QuantizedWeight

    getter rows : Int32
    getter cols : Int32
    getter ggml_type : GGUF::GGMLType

    @host_ptr : Pointer(UInt8)
    @byte_size : UInt64
    # Set when the bytes are NOT mmap-backed but GC-allocated (the Q/gate split builds its buffers
    # by copying raw rows). Holding the slice keeps them alive for the matrix's lifetime; without it
    # only the raw pointer would reference them.
    @owned : Bytes?

    def initialize(@rows, @cols, @ggml_type, @host_ptr, @byte_size, @owned : Bytes? = nil)
      unless @ggml_type == GGUF::GGMLType::Q4_K || @ggml_type == GGUF::GGMLType::Q6_K
        raise ArgumentError.new("GGUFHostMatrix: unsupported type #{@ggml_type}")
      end
    end

    def device_bytes : UInt64
      0_u64
    end

    # ── CudaMatrix interface (for layers that pass device activations) ──
    def gemv(x : CudaMatrix) : CudaMatrix
      result = CudaMatrix.new(x.rows, @cols)
      gemv_into(x, result)
    end

    # Above this many rows, send the weight to the GPU and compute there instead of on the CPU.
    #
    # At one row a host weight is the right call: it streams at DDR5 bandwidth and the CPU is not the
    # limit. Past a few dozen rows the same projection becomes compute-bound and the CPU is ~100x off
    # the GPU, so paying the PCIe upload to borrow the card wins by a wide margin.
    #
    # 32 is llama.cpp's threshold -- its scheduler's offload_op hook moves an op whose weights live in
    # a host buffer onto the GPU once the batch dimension reaches a hardcoded minimum of 32 rows.
    #
    # I tried lowering it to 8, reasoning that the upload is a fixed ~4 ms for ffn_gate's 47.8 MB
    # while the CPU cost scales with rows, which put break-even near 5. Measurement disagreed: on a
    # 9-token incremental prefill, offloading took 973 ms against 806 ms for the CPU path. At that
    # few rows the weight stream still dominates and the upload is pure added cost, so the estimate
    # was wrong and 32 stands. Tunable to re-measure the crossover on other hardware.
    DEFAULT_BATCH_DEVICE_MIN_ROWS = 32

    # Read ONCE. Both were ENV lookups evaluated on EVERY call, and a Crystal ENV lookup is a getenv
    # linear scan of the environment plus a String allocation -- paid per matmul, per layer, per
    # token, including for the untaken branch of a debug print.
    @@debug : Bool = ENV["SHAINET_DEBUG"]? == "1"
    @@offload_op : Bool = ENV.fetch("SHAINET_HOST_OFFLOAD_OP", "1") != "0"

    @@batch_device_min_rows : Int32?

    def self.batch_device_min_rows : Int32
      rows = @@batch_device_min_rows
      return rows if rows
      env = ENV["SHAINET_HOST_OFFLOAD_MIN_ROWS"]?
      rows = (env.try(&.to_i?) || DEFAULT_BATCH_DEVICE_MIN_ROWS)
      @@batch_device_min_rows = rows
      rows
    end

    def gemv_into(x : CudaMatrix, result : CudaMatrix) : CudaMatrix
      if @@debug
        STDERR.puts "  [gguf host gemv] #{@ggml_type} M=#{x.rows} N=#{@cols} K=#{@rows} bytes=#{@byte_size}"
      end

      m = x.rows
      k = @rows
      n = @cols

      if m >= GGUFHostMatrix.batch_device_min_rows && @@offload_op
        if staged = GGUFMatrix.stage_host(k, n, @ggml_type, @host_ptr, @byte_size)
          return staged.gemm_into(x, result)
        end
      end

      # Read activation from device to host
      x_host = Pointer(Float32).malloc(m * k)
      x_dptr = x.device_ptr
      if x_dptr && !x_dptr.null?
        CUDA.memcpy(x_host.as(Pointer(Void)), x_dptr.as(Pointer(Void)),
          (m * k * 4).to_u64, CUDA::MemcpyKind::DeviceToHost)
      else
        x_host.copy_from(x.raw_data.to_unsafe, m * k)
      end

      # Allocate host result
      y_host = Pointer(Float32).malloc(m * n)

      # Run CPU GEMV (AVX2 or scalar fallback)
      used_avx2 = case @ggml_type
                  when .q4_k?
                    CPUKernels.gemv_q4k(x_host, @host_ptr, y_host, m, n, k)
                  when .q6_k?
                    CPUKernels.gemv_q6k(x_host, @host_ptr, y_host, m, n, k)
                  else
                    false
                  end

      unless used_avx2
        gemv_scalar(x_host, y_host, m, k, n)
      end

      # Upload result to device
      r_dptr = result.device_ptr
      if r_dptr && !r_dptr.null?
        CUDA.memcpy(r_dptr.as(Pointer(Void)), y_host.as(Pointer(Void)),
          (m * n * 4).to_u64, CUDA::MemcpyKind::HostToDevice)
        result.mark_device_dirty!
      else
        result.raw_data.to_unsafe.copy_from(y_host, m * n)
      end

      result
    end

    def free!
    end

    # ── Crystal scalar fallback ──
    private def gemv_scalar(x : Pointer(Float32), y : Pointer(Float32),
                            m : Int32, k : Int32, n : Int32)
      case @ggml_type
      when .q4_k?
        gemv_q4k_scalar(x, y, m, k, n)
      when .q6_k?
        gemv_q6k_scalar(x, y, m, k, n)
      end
    end

    private def gemv_q4k_scalar(x : Pointer(Float32), y : Pointer(Float32),
                                m : Int32, k : Int32, n : Int32)
      bs = 144
      vals_per_block = 256
      blocks_per_row = (k + vals_per_block - 1) // vals_per_block
      bytes_per_row = blocks_per_row.to_i64 * bs

      n.times do |col|
        wrow = @host_ptr + col.to_i64 * bytes_per_row
        m.times do |row|
          xrow = x + row * k
          dot = 0.0_f32
          blocks_per_row.times do |blk|
            block = wrow + blk * bs
            d = HFLoader.half_to_f32(block[0].to_u16 | (block[1].to_u16 << 8))
            dmin = HFLoader.half_to_f32(block[2].to_u16 | (block[3].to_u16 << 8))
            scales = block + 4
            qs = block + 16
            base_k = blk * vals_per_block

            4.times do |j64|
              sc0, m0 = HFLoader.get_scale_min_k4_host(j64 * 2, scales)
              sc1, m1 = HFLoader.get_scale_min_k4_host(j64 * 2 + 1, scales)
              d1 = d * sc0.to_f32
              m1v = dmin * m0.to_f32
              d2 = d * sc1.to_f32
              m2v = dmin * m1.to_f32
              32.times do |l|
                ki = base_k + j64 * 64 + l
                break if ki >= k
                dot += xrow[ki] * (d1 * (qs[j64 * 32 + l] & 0xF).to_f32 - m1v)
              end
              32.times do |l|
                ki = base_k + j64 * 64 + l + 32
                break if ki >= k
                dot += xrow[ki] * (d2 * (qs[j64 * 32 + l] >> 4).to_f32 - m2v)
              end
            end
          end
          y[row * n + col] = dot
        end
      end
    end

    private def gemv_q6k_scalar(x : Pointer(Float32), y : Pointer(Float32),
                                m : Int32, k : Int32, n : Int32)
      bs = 210
      vals_per_block = 256
      blocks_per_row = (k + vals_per_block - 1) // vals_per_block
      bytes_per_row = blocks_per_row.to_i64 * bs

      n.times do |col|
        wrow = @host_ptr + col.to_i64 * bytes_per_row
        m.times do |row|
          xrow = x + row * k
          dot = 0.0_f32
          blocks_per_row.times do |blk|
            block = wrow + blk * bs
            ql = block
            qh = block + 128
            sc = block + 192
            d = HFLoader.half_to_f32(block[208].to_u16 | (block[209].to_u16 << 8))
            base_k = blk * vals_per_block

            2.times do |chunk|
              ql_c = ql + chunk * 64
              qh_c = qh + chunk * 32
              sc_c = sc + chunk * 8
              32.times do |l|
                is = l // 16
                k0 = base_k + chunk * 128 + l
                q1 = ((ql_c[l] & 0xF) | (((qh_c[l] >> 0) & 3) << 4)).to_i8! - 32
                q2 = ((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)).to_i8! - 32
                q3 = ((ql_c[l] >> 4) | (((qh_c[l] >> 4) & 3) << 4)).to_i8! - 32
                q4 = ((ql_c[l + 32] >> 4) | (((qh_c[l] >> 6) & 3) << 4)).to_i8! - 32
                s0 = d * sc_c[is].to_i8!.to_f32
                s2 = d * (sc_c + is + 2).value.to_i8!.to_f32
                s4 = d * (sc_c + is + 4).value.to_i8!.to_f32
                s6 = d * (sc_c + is + 6).value.to_i8!.to_f32
                dot += xrow[k0] * s0 * q1.to_f32 if k0 < k
                dot += xrow[k0 + 32] * s2 * q2.to_f32 if k0 + 32 < k
                dot += xrow[k0 + 64] * s4 * q3.to_f32 if k0 + 64 < k
                dot += xrow[k0 + 96] * s6 * q4.to_f32 if k0 + 96 < k
              end
            end
          end
          y[row * n + col] = dot
        end
      end
    end
  end
end
