require "../basic/matrix_layer"

module SHAInet
  # TransformerBlock implements multi-head self-attention followed by a
  # position-wise feed forward network. LayerNorm and dropout are applied
  # with residual connections around each sub layer.
  class TransformerBlock < MatrixLayer
    getter mha : MultiHeadAttention
    getter ffn : PositionWiseFF
    getter norm1 : LayerNorm
    getter norm2 : LayerNorm
    property positional_encoding : SimpleMatrix?
    property drop_percent : Int32

    def initialize(d_model : Int32, num_heads : Int32, ff_hidden : Int32,
                   drop_percent : Int32 = 0)
      super("memory", d_model, SHAInet.none)
      @mha = MultiHeadAttention.new(d_model, num_heads)
      @ffn = PositionWiseFF.new(d_model, ff_hidden)
      @norm1 = LayerNorm.new(d_model)
      @norm2 = LayerNorm.new(d_model)
      @positional_encoding = nil
      @drop_percent = drop_percent
    end

    # `x` is a matrix where each row is a batch element. Sequences can be
    # processed step-by-step using this batch-first layout.
    def forward(x : SimpleMatrix, pe : SimpleMatrix? = nil, mask : SimpleMatrix? = nil)
      input = if enc = (pe || @positional_encoding)
                # Check dimensions and provide better error message
                if enc.cols != x.cols
                  raise "positional encoding feature dimension mismatch: expected d_model=#{x.cols}, got #{enc.cols}"
                end

                # Create a position encoding matrix that matches the input sequence length
                actual_pe = if x.rows <= enc.rows
                              # Use first x.rows positions from the positional encoding
                              SHAInet::SimpleMatrix.new(x.rows, x.cols).tap do |pe_subset|
                                x.rows.times do |i|
                                  x.cols.times do |j|
                                    pe_subset[i, j] = enc[i, j]
                                  end
                                end
                              end
                            else
                              # Need more positions than available - use what we have and pad with zeros
                              SHAInet::SimpleMatrix.new(x.rows, x.cols).tap do |pe_extended|
                                x.rows.times do |i|
                                  x.cols.times do |j|
                                    pe_extended[i, j] = i < enc.rows ? enc[i, j] : 0.0
                                  end
                                end
                              end
                            end
                x + actual_pe
              else
                x
              end
      attn = @mha.forward(input, mask)
      attn = TransformerDropout.apply(attn, @drop_percent) if @drop_percent > 0
      attn = attn + input
      normed = @norm1.forward(attn)
      ff = @ffn.forward(normed)
      ff = TransformerDropout.apply(ff, @drop_percent) if @drop_percent > 0
      ff = ff + normed
      @norm2.forward(ff)
    end

    # `d_out` follows the batch-first convention used by `forward`.
    def backward(d_out : SimpleMatrix)
      d_norm2 = @norm2.backward(d_out)
      d_ff = @ffn.backward(d_norm2)
      d_ff += d_norm2
      d_norm1 = @norm1.backward(d_ff)
      d_attn = @mha.backward(d_norm1)
      d_attn + d_norm1
    end

    def apply_gradients(lr : Float64)
      @ffn.apply_gradients(lr)
      @mha.apply_gradients(lr)
      @norm1.apply_gradients(lr)
      @norm2.apply_gradients(lr)
    end

    # Override MatrixLayer's update_weights to prevent conflicts
    # TransformerBlock manages its own weight updates through apply_gradients
    def update_weights(learning_rate : Float64)
      # No-op: weights are updated via apply_gradients in update_transformer_layers
    end

    def zero_gradients
      @ffn.zero_gradients
      @mha.zero_gradients
      @norm1.zero_gradients
      @norm2.zero_gradients
    end
  end

  alias TransformerLayer = TransformerBlock
end
