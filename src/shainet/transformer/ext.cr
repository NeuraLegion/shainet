module SHAInet
  class MultiHeadAttention
    setter w_q : SimpleMatrix
    setter w_k : SimpleMatrix
    setter w_v : SimpleMatrix
    setter w_o : SimpleMatrix
  end

  class PositionWiseFF
    setter w1 : SimpleMatrix
    setter b1 : SimpleMatrix
    setter w2 : SimpleMatrix
    setter b2 : SimpleMatrix
  end

  class LayerNorm
    setter gamma : SimpleMatrix
    setter beta : SimpleMatrix
  end
end
