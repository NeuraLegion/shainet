module SHAInet
  class TransformerLayer < Layer
    getter mha : MultiHeadAttention
    getter ffn : PositionWiseFF
    property positional_encoding : SimpleMatrix?

    def initialize(d_model : Int32, num_heads : Int32, ff_hidden : Int32)
      super("memory", d_model, SHAInet.none)
      @mha = MultiHeadAttention.new(d_model, num_heads)
      @ffn = PositionWiseFF.new(d_model, ff_hidden)
      @positional_encoding = nil
    end

    def forward(x : SimpleMatrix, pe : SimpleMatrix? = nil)
      input = if enc = (pe || @positional_encoding)
                raise "positional encoding size mismatch" unless enc.rows == x.rows && enc.cols == x.cols
                x + enc
              else
                x
              end
      attn_out = @mha.forward(input)
      ff_out = @ffn.forward(attn_out)
      ff_out
    end

    def backward(d_out : SimpleMatrix)
      d_attn = @ffn.backward(d_out)
      d_in = @mha.backward(d_attn)
      d_in
    end

    def apply_gradients(lr : Float64)
      @ffn.apply_gradients(lr)
      @mha.apply_gradients(lr)
    end

    def zero_gradients
      @ffn.zero_gradients
      @mha.zero_gradients
    end

    def neurons
      [] of Neuron
    end
  end
end
