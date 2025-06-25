module SHAInet
  class MultiHeadAttention
    def w_q=(m : TensorMatrix)
      @w_q = m
    end

    def w_k=(m : TensorMatrix)
      @w_k = m
    end

    def w_v=(m : TensorMatrix)
      @w_v = m
    end

    def w_o=(m : TensorMatrix)
      @w_o = m
    end
  end

  class PositionWiseFF
    def w1=(m : SimpleMatrix)
      @w1 = m
    end

    def b1=(m : SimpleMatrix)
      @b1 = m
    end

    def w2=(m : SimpleMatrix)
      @w2 = m
    end

    def b2=(m : SimpleMatrix)
      @b2 = m
    end
  end

  class LayerNorm
    def gamma=(m : SimpleMatrix)
      @gamma = m
    end

    def beta=(m : SimpleMatrix)
      @beta = m
    end
  end
end
