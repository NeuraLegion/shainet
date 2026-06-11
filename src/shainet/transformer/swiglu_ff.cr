module SHAInet
  # SwiGLU Feed-Forward Network as used in LLaMA/Mistral.
  # Formula: output = down_proj(silu(gate_proj(x)) * up_proj(x))
  class SwiGLUFF
    property gate_proj : SimpleMatrix | CudaMatrix | QuantizedCudaMatrix
    property up_proj : SimpleMatrix | CudaMatrix | QuantizedCudaMatrix
    property down_proj : SimpleMatrix | CudaMatrix | QuantizedCudaMatrix

    def initialize(d_model : Int32, ff_hidden : Int32)
      @gate_proj = SimpleMatrix.new(d_model, ff_hidden)
      @up_proj = SimpleMatrix.new(d_model, ff_hidden)
      @down_proj = SimpleMatrix.new(ff_hidden, d_model)
    end

    def to_gpu!(quantize : Bool = false)
      return unless CUDA.fully_available?
      if quantize
        @gate_proj = QuantizedCudaMatrix.from_simple(@gate_proj.as(SimpleMatrix)) if @gate_proj.is_a?(SimpleMatrix)
        @up_proj = QuantizedCudaMatrix.from_simple(@up_proj.as(SimpleMatrix)) if @up_proj.is_a?(SimpleMatrix)
        @down_proj = QuantizedCudaMatrix.from_simple(@down_proj.as(SimpleMatrix)) if @down_proj.is_a?(SimpleMatrix)
      else
        @gate_proj = @gate_proj.as(SimpleMatrix).to_cuda unless @gate_proj.is_a?(CudaMatrix)
        @up_proj = @up_proj.as(SimpleMatrix).to_cuda unless @up_proj.is_a?(CudaMatrix)
        @down_proj = @down_proj.as(SimpleMatrix).to_cuda unless @down_proj.is_a?(CudaMatrix)
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

    private def matmul(x : SimpleMatrix, w : SimpleMatrix | CudaMatrix | QuantizedCudaMatrix) : SimpleMatrix
      if w.is_a?(QuantizedCudaMatrix)
        x_gpu = CudaMatrix.new(x.rows, x.cols)
        x.rows.times { |r| x.cols.times { |c| x_gpu[r, c] = x[r, c] } }
        x_gpu.sync_to_device!("q8_ffn_in")
        result_gpu = w.gemv(x_gpu)
        result_gpu.sync_from_device!("q8_ffn_out") if result_gpu.device_dirty?
        result = SimpleMatrix.new(result_gpu.rows, result_gpu.cols)
        result_gpu.rows.times { |r| result_gpu.cols.times { |c| result[r, c] = result_gpu[r, c].to_f32 } }
        result
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
