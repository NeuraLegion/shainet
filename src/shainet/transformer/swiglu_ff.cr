module SHAInet
  # SwiGLU Feed-Forward Network as used in LLaMA/Mistral.
  # Formula: output = down_proj(silu(gate_proj(x)) * up_proj(x))
  class SwiGLUFF
    property gate_proj : SimpleMatrix | CudaMatrix
    property up_proj : SimpleMatrix | CudaMatrix
    property down_proj : SimpleMatrix | CudaMatrix

    def initialize(d_model : Int32, ff_hidden : Int32)
      @gate_proj = SimpleMatrix.new(d_model, ff_hidden)
      @up_proj = SimpleMatrix.new(d_model, ff_hidden)
      @down_proj = SimpleMatrix.new(ff_hidden, d_model)
    end

    def to_gpu!
      return unless CUDA.fully_available?
      @gate_proj = @gate_proj.as(SimpleMatrix).to_cuda.tap(&.mark_device_dirty!) unless @gate_proj.is_a?(CudaMatrix)
      @up_proj = @up_proj.as(SimpleMatrix).to_cuda.tap(&.mark_device_dirty!) unless @up_proj.is_a?(CudaMatrix)
      @down_proj = @down_proj.as(SimpleMatrix).to_cuda.tap(&.mark_device_dirty!) unless @down_proj.is_a?(CudaMatrix)
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

    private def matmul(x : SimpleMatrix, w : SimpleMatrix | CudaMatrix) : SimpleMatrix
      if w.is_a?(CudaMatrix)
        n = x.rows * x.cols
        x_gpu = CudaMatrix.new(x.rows, x.cols)
        x_gpu.raw_data.to_unsafe.copy_from(x.data.to_unsafe, n)
        x_gpu.sync_to_device!("ffn_in")
        result_gpu = x_gpu * w
        result_gpu.sync_from_device!("ffn_out") if result_gpu.device_dirty?
        result = SimpleMatrix.new(result_gpu.rows, result_gpu.cols)
        result.data.to_unsafe.copy_from(result_gpu.raw_data.to_unsafe, result.rows * result.cols)
        result
      else
        x * w
      end
    end
  end
end
