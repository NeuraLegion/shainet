module SHAInet
  # A host-matrix input times a weight that may be fp32, on-device fp32, or quantized.
  #
  # Extracted from LlamaBlock so GatedDeltaNetBlock can reach the same path. Without it a hybrid
  # stack quantizes only its attention layers: on Qwen3.5-9B that is 8 of 32 layers, leaving the
  # other 24 in fp32 and the model too large to load at all. Both block types now dispatch here,
  # so there is one implementation of the decode fast path rather than two that drift.
  #
  # The three arms exist because they have genuinely different costs:
  #
  #   QuantizedWeight - dequantized inside the GEMM kernel. At M=1 (decode) it reuses persistent
  #                     device buffers keyed by width, because allocating and freeing a pair of
  #                     device matrices per projection per layer per token dominated the step.
  #   CudaMatrix      - resident fp32, straight cuBLAS SGEMM.
  #   SimpleMatrix    - host fallback, which is also the CPU-only build's only path.
  #
  # The buffer hashes are keyed by matrix width, and a transformer has a small fixed set of
  # widths, so the count is bounded by the architecture rather than by how long the process runs.
  module QuantizedProjection
    @q8_in_bufs = Hash(Int32, CudaMatrix).new
    @q8_out_bufs = Hash(Int32, CudaMatrix).new

    # Quantize a weight to the requested width: 4 -> Q4, 8 -> Q8. With `offload` the (Q4-only)
    # weight stays in host RAM as a Q4HostMatrix and is streamed on demand. An already-quantized
    # weight is returned untouched: quantization happens once during load, and switching format
    # in place would silently double the peak it exists to avoid.
    def to_quant(w : SimpleMatrix | CudaMatrix | QuantizedWeight, bits : Int32, offload : Bool = false) : QuantizedWeight
      raise ArgumentError.new("unsupported quantization bits: #{bits} (expected 8 or 4)") unless bits == 8 || bits == 4
      raise ArgumentError.new("dense offload currently supports 4-bit only (got #{bits}-bit)") if offload && bits != 4
      case w
      when QuantizedWeight then w
      when CudaMatrix
        sm = w.to_simple
        offload ? Q4HostMatrix.from_simple(sm) : (bits == 4 ? Q4CudaMatrix.from_simple(sm) : QuantizedCudaMatrix.from_simple(sm))
      else
        sm = w.as(SimpleMatrix)
        offload ? Q4HostMatrix.from_simple(sm) : (bits == 4 ? Q4CudaMatrix.from_simple(sm) : QuantizedCudaMatrix.from_simple(sm))
      end
    end

    def gpu_matmul(x : SimpleMatrix, w : SimpleMatrix | CudaMatrix | QuantizedWeight) : SimpleMatrix
      if w.is_a?(QuantizedWeight)
        if x.rows == 1
          # Decode (M=1): reuse persistent device buffers, no per-call alloc/free.
          xb = (@q8_in_bufs[x.cols] ||= CudaMatrix.new(1, x.cols))
          Profile.measure("gemm.in_h2d") do
            xb.raw_data.to_unsafe.copy_from(x.data.to_unsafe, x.cols)
            xb.mark_host_modified!
            xb.sync_to_device!("q8_gemm_in")
          end
          ob = (@q8_out_bufs[w.cols] ||= CudaMatrix.new(1, w.cols))
          Profile.measure("gemm.kernel") { w.gemv_into(xb, ob) }
          Profile.measure("gemm.out_d2h") { ob.sync_from_device!("q8_gemm_out") if ob.device_dirty? }
          result = SimpleMatrix.new(1, w.cols)
          Profile.measure("gemm.result_copy") do
            result.data.to_unsafe.copy_from(ob.raw_data.to_unsafe, w.cols)
          end
          result
        else
          # Prefill / batch (M>1): one-off allocation.
          x_gpu = CudaMatrix.new(x.rows, x.cols)
          x_gpu.raw_data.to_unsafe.copy_from(x.data.to_unsafe, x.rows * x.cols)
          x_gpu.sync_to_device!("q8_gemm_in")
          result_gpu = w.gemv(x_gpu)
          result_gpu.sync_from_device!("q8_gemm_out") if result_gpu.device_dirty?
          result = SimpleMatrix.new(result_gpu.rows, result_gpu.cols)
          result.data.to_unsafe.copy_from(result_gpu.raw_data.to_unsafe, result_gpu.rows * result_gpu.cols)
          x_gpu.free!
          result_gpu.free!
          result
        end
      elsif w.is_a?(CudaMatrix)
        x_gpu = CudaMatrix.new(x.rows, x.cols)
        # Bulk copy, not element-by-element. The quantized branch above already did this; this one
        # walked `x_gpu[r, c] = x[r, c]`, which for a [1216, 4096] activation is 5M scalar accessor
        # calls. That made the Gated DeltaNet gate projections cost 0.329 s per layer -- more than the
        # recurrence they feed -- for a GEMM cuBLAS finishes in milliseconds.
        x_gpu.raw_data.to_unsafe.copy_from(x.data.to_unsafe, x.rows * x.cols)
        x_gpu.mark_host_modified!
        x_gpu.sync_to_device!("gemm_in")
        result_gpu = x_gpu * w # cuBLAS SGEMM
        result_gpu.sync_from_device!("gemm_out") if result_gpu.device_dirty?
        result = SimpleMatrix.new(result_gpu.rows, result_gpu.cols)
        result.data.to_unsafe.copy_from(result_gpu.raw_data.to_unsafe, result_gpu.rows * result_gpu.cols)
        x_gpu.free!
        result_gpu.free!
        result
      else
        # Host fp32 matmul -- use AVX2+OpenMP C kernel when available
        m = x.rows
        k = x.cols
        n = w.cols
        if CPUKernels.available? && m * n > 1024
          c_ptr = Pointer(Float32).malloc(m * n)
          CPUKernels.sgemm(
            x.data.to_unsafe, w.data.to_unsafe, c_ptr,
            m, n, k,
          )
          result = SimpleMatrix.new(m, n)
          result.data.to_unsafe.copy_from(c_ptr, m * n)
          result
        else
          x * w
        end
      end
    end

    # Release the cached decode buffers. A block moved back to host keeps no device memory.
    def free_projection_buffers!
      @q8_in_bufs.each_value(&.free!)
      @q8_out_bufs.each_value(&.free!)
      @q8_in_bufs.clear
      @q8_out_bufs.clear
    end
  end
end
