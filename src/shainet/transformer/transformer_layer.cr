module SHAInet
  class TransformerLayer < Layer
    getter mha : MultiHeadAttention
    getter ffn : PositionWiseFF
    getter norm1 : LayerNorm
    getter norm2 : LayerNorm
    property positional_encoding : SimpleMatrix?

    def initialize(d_model : Int32, num_heads : Int32, ff_hidden : Int32)
      super("memory", d_model, SHAInet.none)
      @mha = MultiHeadAttention.new(d_model, num_heads)
      @ffn = PositionWiseFF.new(d_model, ff_hidden)
      @norm1 = LayerNorm.new(d_model)
      @norm2 = LayerNorm.new(d_model)
      @positional_encoding = nil
    end

    def forward(x : SimpleMatrix, pe : SimpleMatrix? = nil, mask : SimpleMatrix? = nil)
      input = if enc = (pe || @positional_encoding)
                raise "positional encoding size mismatch" unless enc.rows == x.rows && enc.cols == x.cols
                x + enc
              else
                x
              end
      attn_out = @mha.forward(input, mask)
      normed = @norm1.forward(attn_out)
      ff_out = @ffn.forward(normed)
      @norm2.forward(ff_out)
    end

    def backward(d_out : SimpleMatrix)
      d_norm2 = @norm2.backward(d_out)
      d_ff = @ffn.backward(d_norm2)
      d_norm1 = @norm1.backward(d_ff)
      @mha.backward(d_norm1)
    end

    def apply_gradients(lr : Float64)
      @ffn.apply_gradients(lr)
      @mha.apply_gradients(lr)
      @norm1.apply_gradients(lr)
      @norm2.apply_gradients(lr)
    end

    def zero_gradients
      @ffn.zero_gradients
      @mha.zero_gradients
      @norm1.zero_gradients
      @norm2.zero_gradients
    end

    def neurons
      [] of Neuron
    end
  end
end
