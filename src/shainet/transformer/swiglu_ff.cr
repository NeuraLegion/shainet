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
      @gate_proj = @gate_proj.as(SimpleMatrix).to_cuda unless @gate_proj.is_a?(CudaMatrix)
      @up_proj = @up_proj.as(SimpleMatrix).to_cuda unless @up_proj.is_a?(CudaMatrix)
      @down_proj = @down_proj.as(SimpleMatrix).to_cuda unless @down_proj.is_a?(CudaMatrix)
    end

    def forward(x : SimpleMatrix) : SimpleMatrix
      gate = x * @gate_proj.as(SimpleMatrix)
      up = x * @up_proj.as(SimpleMatrix)

      rows = gate.rows
      cols = gate.cols
      hidden = SimpleMatrix.new(rows, cols)
      rows.times do |i|
        cols.times do |j|
          g = gate[i, j]
          hidden[i, j] = (g / (1.0 + Math.exp(-g))) * up[i, j]
        end
      end

      hidden * @down_proj.as(SimpleMatrix)
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
  end
end
