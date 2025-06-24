module SHAInet
  # Generates sinusoidal positional encodings as described in "Attention is All You Need".
  # Returns a SimpleMatrix of shape (max_len, d_model).
  class PositionalEncoding
    def self.sinusoidal(max_len : Int32, d_model : Int32) : SimpleMatrix
      pe = SimpleMatrix.new(max_len, d_model)
      max_len.times do |pos|
        d_model.times do |i|
          div_term = 1.0 / (10000.0 ** ((2 * (i // 2)).to_f64 / d_model))
          angle = pos.to_f64 * div_term
          if i.even?
            pe[pos, i] = Math.sin(angle)
          else
            pe[pos, i] = Math.cos(angle)
          end
        end
      end
      pe
    end
  end
end
