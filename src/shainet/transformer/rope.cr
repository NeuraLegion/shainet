module SHAInet
  # Rotary Positional Embeddings (RoPE) as used in HuggingFace LLaMA/Mistral.
  # Uses the half-split (GPT-NeoX) convention: element i is paired with
  # element i + head_dim/2, both rotated by position * theta^(-2i/head_dim).
  module RoPE
    # Apply RoPE to a matrix of shape [seq_len, head_dim].
    def self.apply(x : SimpleMatrix, start_pos : Int32 = 0, theta : Float64 = 10000.0) : SimpleMatrix
      seq_len = x.rows
      head_dim = x.cols
      half = head_dim // 2
      result = SimpleMatrix.new(seq_len, head_dim)

      seq_len.times do |pos|
        actual_pos = pos + start_pos
        half.times do |i|
          freq = 1.0 / (theta ** (2.0 * i / head_dim))
          angle = actual_pos * freq
          cos_val = Math.cos(angle)
          sin_val = Math.sin(angle)

          x0 = x[pos, i]
          x1 = x[pos, i + half]
          result[pos, i] = x0 * cos_val - x1 * sin_val
          result[pos, i + half] = x1 * cos_val + x0 * sin_val
        end
        # Copy through any odd trailing dimension unchanged
        if head_dim.odd?
          result[pos, head_dim - 1] = x[pos, head_dim - 1]
        end
      end

      result
    end
  end
end
