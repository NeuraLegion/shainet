module SHAInet
  # SwiGLU Feed-Forward Network as used in LLaMA/Mistral.
  # Formula: output = down_proj(silu(gate_proj(x)) * up_proj(x))
  # Where silu(x) = x * sigmoid(x)
  class SwiGLUFF
    property gate_proj : SimpleMatrix | CudaMatrix # [d_model, ff_hidden]
    property up_proj : SimpleMatrix | CudaMatrix   # [d_model, ff_hidden]
    property down_proj : SimpleMatrix | CudaMatrix # [ff_hidden, d_model]

    def initialize(d_model : Int32, ff_hidden : Int32)
      @gate_proj = SimpleMatrix.new(d_model, ff_hidden)
      @up_proj = SimpleMatrix.new(d_model, ff_hidden)
      @down_proj = SimpleMatrix.new(ff_hidden, d_model)
    end

    def forward(x : SimpleMatrix) : SimpleMatrix
      gate = x * @gate_proj.as(SimpleMatrix) # [seq, ff_hidden]
      up = x * @up_proj.as(SimpleMatrix)     # [seq, ff_hidden]

      # SiLU(gate) * up
      rows = gate.rows
      cols = gate.cols
      hidden = SimpleMatrix.new(rows, cols)
      rows.times do |i|
        cols.times do |j|
          g = gate[i, j]
          silu_g = g * (1.0 / (1.0 + Math.exp(-g))) # silu = x * sigmoid(x)
          hidden[i, j] = silu_g * up[i, j]
        end
      end

      # down_proj
      hidden * @down_proj.as(SimpleMatrix) # [seq, d_model]
    end
  end
end
