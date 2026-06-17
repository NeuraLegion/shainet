module SHAInet
  # SwiGLU Feed-Forward Network as used in LLaMA/Mistral.
  # Formula: output = down_proj(silu(gate_proj(x)) * up_proj(x))
  class SwiGLUFF
    property gate_proj : SimpleMatrix | CudaMatrix | QuantizedWeight
    property up_proj : SimpleMatrix | CudaMatrix | QuantizedWeight
    property down_proj : SimpleMatrix | CudaMatrix | QuantizedWeight

    # allocate: when false, weights start as empty 0x0 placeholders instead of
    # full [d_model, ff_hidden] matrices. Used for MoE experts, which are always
    # overwritten by the loader — avoids allocating hundreds of fp32 expert
    # matrices up front (a 48-layer/128-expert model would otherwise need ~100GB
    # before streaming quantization ever runs).
    def initialize(d_model : Int32, ff_hidden : Int32, allocate : Bool = true)
      if allocate
        @gate_proj = SimpleMatrix.new(d_model, ff_hidden)
        @up_proj = SimpleMatrix.new(d_model, ff_hidden)
        @down_proj = SimpleMatrix.new(ff_hidden, d_model)
      else
        @gate_proj = SimpleMatrix.new(0, 0)
        @up_proj = SimpleMatrix.new(0, 0)
        @down_proj = SimpleMatrix.new(0, 0)
      end
    end

    # Persistent single-row GEMV workspaces for decode (M=1), keyed by width.
    @q8_in_bufs = Hash(Int32, CudaMatrix).new
    @q8_out_bufs = Hash(Int32, CudaMatrix).new

    def to_gpu!(quantize : Bool = false, bits : Int32 = 8)
      return unless CUDA.fully_available?
      # Catch placeholder experts (allocate: false) that were never loaded — promoting
      # or quantizing a 0x0 matrix would otherwise surface as a cryptic CUDA malloc(0)
      # failure.
      if (g = @gate_proj).is_a?(SimpleMatrix) && (g.rows == 0 || g.cols == 0)
        raise "SwiGLUFF weights are uninitialized (0x0); load weights before calling to_gpu!"
      end
      if quantize
        @gate_proj = to_quant(@gate_proj, bits)
        @up_proj = to_quant(@up_proj, bits)
        @down_proj = to_quant(@down_proj, bits)
      else
        # Only promote host weights; leave existing CudaMatrix/QuantizedWeight as-is.
        @gate_proj = @gate_proj.as(SimpleMatrix).to_cuda if @gate_proj.is_a?(SimpleMatrix)
        @up_proj = @up_proj.as(SimpleMatrix).to_cuda if @up_proj.is_a?(SimpleMatrix)
        @down_proj = @down_proj.as(SimpleMatrix).to_cuda if @down_proj.is_a?(SimpleMatrix)
      end
    end

    # Quantize a weight to the requested bit width: bits == 4 -> Q4, bits == 8 ->
    # Q8. Already-quantized weights are returned unchanged.
    private def to_quant(w : SimpleMatrix | CudaMatrix | QuantizedWeight, bits : Int32) : QuantizedWeight
      raise ArgumentError.new("unsupported quantization bits: #{bits} (expected 8 or 4)") unless bits == 8 || bits == 4
      case w
      when QuantizedWeight then w
      when CudaMatrix
        sm = w.to_simple
        bits == 4 ? Q4CudaMatrix.from_simple(sm) : QuantizedCudaMatrix.from_simple(sm)
      else
        sm = w.as(SimpleMatrix)
        bits == 4 ? Q4CudaMatrix.from_simple(sm) : QuantizedCudaMatrix.from_simple(sm)
      end
    end

    def forward(x : SimpleMatrix) : SimpleMatrix
      gate = matmul(x, @gate_proj)
      up = matmul(x, @up_proj)

      rows = gate.rows
      cols = gate.cols
      hidden = SimpleMatrix.new(rows, cols)
      rows.times do |i|
        cols.times do |j|
          g = gate[i, j]
          hidden[i, j] = (g / (1.0 + Math.exp(-g))) * up[i, j]
        end
      end

      matmul(hidden, @down_proj)
    end

    def forward(x : CudaMatrix) : CudaMatrix
      gate = x * @gate_proj.as(CudaMatrix) # cuBLAS GEMM
      up = x * @up_proj.as(CudaMatrix)     # cuBLAS GEMM

      # SiLU element-wise on CPU (no custom kernel yet)
      gate.sync_from_device!("swiglu") if gate.device_dirty?
      up.sync_from_device!("swiglu") if up.device_dirty?

      rows = gate.rows
      cols = gate.cols
      hidden = CudaMatrix.new(rows, cols)
      rows.times do |i|
        cols.times do |j|
          g = gate[i, j]
          hidden[i, j] = (g / (1.0 + Math.exp(-g))) * up[i, j]
        end
      end
      hidden.sync_to_device!("swiglu_done")

      hidden * @down_proj.as(CudaMatrix) # cuBLAS GEMM
    end

    private def matmul(x : SimpleMatrix, w : SimpleMatrix | CudaMatrix | QuantizedWeight) : SimpleMatrix
      if w.is_a?(QuantizedWeight)
        if x.rows == 1
          xb = (@q8_in_bufs[x.cols] ||= CudaMatrix.new(1, x.cols))
          xb.raw_data.to_unsafe.copy_from(x.data.to_unsafe, x.cols)
          xb.mark_host_modified!
          xb.sync_to_device!("q8_ffn_in")
          ob = (@q8_out_bufs[w.cols] ||= CudaMatrix.new(1, w.cols))
          w.gemv_into(xb, ob)
          ob.sync_from_device!("q8_ffn_out") if ob.device_dirty?
          result = SimpleMatrix.new(1, w.cols)
          result.data.to_unsafe.copy_from(ob.raw_data.to_unsafe, w.cols)
          result
        else
          x_gpu = CudaMatrix.new(x.rows, x.cols)
          x_gpu.raw_data.to_unsafe.copy_from(x.data.to_unsafe, x.rows * x.cols)
          x_gpu.sync_to_device!("q8_ffn_in")
          result_gpu = w.gemv(x_gpu)
          result_gpu.sync_from_device!("q8_ffn_out") if result_gpu.device_dirty?
          result = SimpleMatrix.new(result_gpu.rows, result_gpu.cols)
          result.data.to_unsafe.copy_from(result_gpu.raw_data.to_unsafe, result_gpu.rows * result_gpu.cols)
          result
        end
      elsif w.is_a?(CudaMatrix)
        x_gpu = CudaMatrix.new(x.rows, x.cols)
        x.rows.times { |r| x.cols.times { |c| x_gpu[r, c] = x[r, c] } }
        x_gpu.sync_to_device!("ffn_in")
        result_gpu = x_gpu * w
        result_gpu.sync_from_device!("ffn_out") if result_gpu.device_dirty?
        result = SimpleMatrix.new(result_gpu.rows, result_gpu.cols)
        result_gpu.rows.times { |r| result_gpu.cols.times { |c| result[r, c] = result_gpu[r, c].to_f32 } }
        result
      else
        x * w
      end
    end
  end
end
