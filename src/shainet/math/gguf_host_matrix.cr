module SHAInet
  # Host-resident GGUF k-quant weight matrix backed by an mmap'd file region.
  # Performs GEMV entirely on the CPU using the raw Q4_K/Q6_K blocks -- no PCIe
  # transfer, no GPU involvement. This is what llama.cpp does for its CPU layers.
  #
  # The activation input comes as a SimpleMatrix (host fp32); the output is a
  # SimpleMatrix. The block's forward path stays on the host for CPU layers.
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

    def device_bytes : UInt64
      0_u64
    end

    # CPU-side GEMV: y[M, N] = x[M, K] * dequant(W[N, K])
    # x is a host SimpleMatrix [M, K], result is [M, N].
    # Dequants each weight value on-the-fly during the dot product.
    def gemv_host(x : SimpleMatrix) : SimpleMatrix
      m = x.rows
      result = SimpleMatrix.new(m, @cols)
      case @ggml_type
      when .q4_k?
        gemv_q4k_host(x, result, m)
      when .q6_k?
        gemv_q6k_host(x, result, m)
      end
      result
    end

    # QuantizedWeight interface: takes CudaMatrix, returns CudaMatrix.
    # For CPU layers the activation arrives as a CudaMatrix from the device chain;
    # we read it back to host, compute on CPU, and upload the result.
    def gemv(x : CudaMatrix) : CudaMatrix
      result = CudaMatrix.new(x.rows, @cols)
      gemv_into(x, result)
    end

    def gemv_into(x : CudaMatrix, result : CudaMatrix) : CudaMatrix
      STDERR.puts "  [gguf host gemv] #{@ggml_type} M=#{x.rows} N=#{@cols} K=#{@rows} bytes=#{@byte_size}" if ENV["SHAINET_DEBUG"]? == "1"
      # Read activation from device to host
      x.sync_from_device!("gguf_host_in") if x.device_dirty?
      host_x = SimpleMatrix.new(x.rows, x.cols)
      host_x.data.to_unsafe.copy_from(x.raw_data.to_unsafe, x.rows * x.cols)

      # CPU GEMV
      host_result = gemv_host(host_x)

      # Upload result to device
      result.raw_data.to_unsafe.copy_from(host_result.data.to_unsafe, host_result.rows * host_result.cols)
      result.mark_host_modified!
      result.sync_to_device!("gguf_host_out")
      result
    end

    def free!
    end

    # Q4_K CPU GEMV: for each output row n, compute dot(x[m,:], dequant(W[n,:]))
    private def gemv_q4k_host(x : SimpleMatrix, result : SimpleMatrix, m : Int32)
      bs = 144 # Q4_K block size in bytes
      vals_per_block = 256
      blocks_per_row = (@rows + vals_per_block - 1) // vals_per_block
      bytes_per_row = blocks_per_row.to_i64 * bs

      @cols.times do |n|
        wrow = @host_ptr + n.to_i64 * bytes_per_row
        m.times do |row|
          xrow = x.data.to_unsafe + row * @rows
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
              m1_val = dmin * m0.to_f32
              d2 = d * sc1.to_f32
              m2_val = dmin * m1.to_f32
              32.times do |l|
                k = base_k + j64 * 64 + l
                break if k >= @rows
                wval = d1 * (qs[j64 * 32 + l] & 0xF).to_f32 - m1_val
                dot += xrow[k] * wval
              end
              32.times do |l|
                k = base_k + j64 * 64 + l + 32
                break if k >= @rows
                wval = d2 * (qs[j64 * 32 + l] >> 4).to_f32 - m2_val
                dot += xrow[k] * wval
              end
            end
          end
          result[row, n] = dot.to_f64
        end
      end
    end

    # Q6_K CPU GEMV
    private def gemv_q6k_host(x : SimpleMatrix, result : SimpleMatrix, m : Int32)
      bs = 210 # Q6_K block size
      vals_per_block = 256
      blocks_per_row = (@rows + vals_per_block - 1) // vals_per_block
      bytes_per_row = blocks_per_row.to_i64 * bs

      @cols.times do |n|
        wrow = @host_ptr + n.to_i64 * bytes_per_row
        m.times do |row|
          xrow = x.data.to_unsafe + row * @rows
          dot = 0.0_f32
          blocks_per_row.times do |blk|
            block = wrow + blk * bs
            ql = block
            qh = block + 128
            sc = block + 192
            d = HFLoader.half_to_f32(block[208].to_u16 | (block[209].to_u16 << 8))
            base_k = blk * vals_per_block

            (vals_per_block // 128).times do |chunk|
              ql_c = ql + chunk * 64
              qh_c = qh + chunk * 32
              sc_c = sc + chunk * 8
              32.times do |l|
                is = l // 16
                k0 = base_k + chunk * 128 + l
                k1 = k0 + 32
                k2 = k0 + 64
                k3 = k0 + 96
                q1 = ((ql_c[l] & 0xF) | (((qh_c[l] >> 0) & 3) << 4)).to_i8! - 32
                q2 = ((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)).to_i8! - 32
                q3 = ((ql_c[l] >> 4) | (((qh_c[l] >> 4) & 3) << 4)).to_i8! - 32
                q4 = ((ql_c[l + 32] >> 4) | (((qh_c[l] >> 6) & 3) << 4)).to_i8! - 32
                s0 = sc_c[is].to_i8!.to_f32
                s2 = sc_c[is + 2].to_i8!.to_f32
                s4 = sc_c[is + 4].to_i8!.to_f32
                s6 = sc_c[is + 6].to_i8!.to_f32
                dot += xrow[k0] * d * s0 * q1.to_f32 if k0 < @rows
                dot += xrow[k1] * d * s2 * q2.to_f32 if k1 < @rows
                dot += xrow[k2] * d * s4 * q3.to_f32 if k2 < @rows
                dot += xrow[k3] * d * s6 * q4.to_f32 if k3 < @rows
              end
            end
          end
          result[row, n] = dot.to_f64
        end
      end
    end
  end
end
