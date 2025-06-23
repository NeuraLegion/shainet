module SHAInet
  class TransformerLayer < Layer
    getter mha : MultiHeadAttention
    getter ffn : PositionWiseFF

    def initialize(d_model : Int32, num_heads : Int32, ff_hidden : Int32)
      super("memory", d_model, SHAInet.none)
      @mha = MultiHeadAttention.new(d_model, num_heads)
      @ffn = PositionWiseFF.new(d_model, ff_hidden)
    end

    def forward(x : SimpleMatrix)
      attn_out = @mha.forward(x)
      ff_out = @ffn.forward(attn_out)
      ff_out
    end

    def backward(d_out : SimpleMatrix, lr : Float64)
      d_attn = @ffn.backward(d_out, lr)
      d_in = @mha.backward(d_attn, lr)
      d_in
    end

    def neurons
      [] of Neuron
    end
  end
end
