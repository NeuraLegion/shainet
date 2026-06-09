module SHAInet
  # Rotary Positional Embeddings (RoPE) as used in LLaMA/Mistral.
  # Applies rotation to pairs of dimensions in Q and K vectors.
  module RoPE
    # Apply RoPE to a matrix of shape [seq_len, head_dim].
    # Each pair of adjacent elements (2i, 2i+1) is rotated by
    # position * theta^(-2i/head_dim).
    def self.apply(x : SimpleMatrix, start_pos : Int32 = 0, theta : Float64 = 10000.0) : SimpleMatrix
      seq_len = x.rows
      head_dim = x.cols
      result = SimpleMatrix.new(seq_len, head_dim)

      seq_len.times do |pos|
        actual_pos = pos + start_pos
        (head_dim // 2).times do |i|
          freq = 1.0 / (theta ** (2.0 * i / head_dim))
          angle = actual_pos * freq
          cos_val = Math.cos(angle)
          sin_val = Math.sin(angle)

          x0 = x[pos, 2 * i]
          x1 = x[pos, 2 * i + 1]
          result[pos, 2 * i] = x0 * cos_val - x1 * sin_val
          result[pos, 2 * i + 1] = x0 * sin_val + x1 * cos_val
        end
      end

      result
    end
  end
end
