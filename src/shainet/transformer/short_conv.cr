module SHAInet
  # Short causal depthwise convolution over the sequence, per channel.
  #
  # Part of the Gated DeltaNet block's q/k/v paths. The paper's ablation is blunt about it:
  # removing the short conv costs more perplexity than removing the output normalization, so it
  # is not decoration.
  #
  # "Depthwise" means each channel is convolved with its own kernel and channels never mix --
  # so this is `channels` independent 1-D convolutions of width `kernel`, not a matmul.
  #
  # Causal, so position t sees only t-kernel+1 .. t. The left edge is zero-padded, which is what
  # makes prefill and decode agree: a fresh sequence starts with an all-zero window, and that is
  # exactly the state a decode run begins from.
  class ShortConv
    getter channels : Int32
    getter kernel : Int32
    getter weight : SimpleMatrix
    getter bias : SimpleMatrix

    # weight is [channels, kernel], laid out so weight[c, j] is the tap j positions back from
    # the current one -- j = 0 is the current position.
    def initialize(@channels : Int32, @kernel : Int32)
      raise ArgumentError.new("channels must be positive") unless @channels > 0
      raise ArgumentError.new("kernel must be positive") unless @kernel > 0
      @weight = SimpleMatrix.new(@channels, @kernel)
      @bias = SimpleMatrix.new(1, @channels)
    end

    # The rolling window a decode step needs.
    #
    # Softmax attention keeps a KV cache that grows with context; this keeps kernel-1 positions
    # and nothing more, whatever the context length. Column 0 is the oldest.
    def new_state : SimpleMatrix
      SimpleMatrix.new(@channels, @kernel - 1, 0.0)
    end

    # Convolve a whole sequence. `x` is [seq, channels].
    #
    # `state` carries the tail of the PREVIOUS call, so a chunked prefill and a token-at-a-time
    # decode produce the same numbers as one whole-sequence call. Passing nil means a fresh
    # sequence, i.e. zero padding on the left. The state is mutated and returned.
    def forward(x : SimpleMatrix, state : SimpleMatrix? = nil) : {SimpleMatrix, SimpleMatrix}
      seq = x.rows
      raise ArgumentError.new("expected #{@channels} channels, got #{x.cols}") unless x.cols == @channels

      s = state || new_state
      raise ArgumentError.new("state must be [#{@channels}, #{@kernel - 1}]") unless s.rows == @channels && s.cols == @kernel - 1

      dst = SimpleMatrix.new(seq, @channels, 0.0)
      seq.times do |t|
        @channels.times do |c|
          acc = @bias[0, c].to_f64
          @kernel.times do |j|
            # j positions back from t: inside this call when t - j >= 0, otherwise from the
            # carried state, whose last column is the most recent past position.
            back = t - j
            val = if back >= 0
                    x[back, c].to_f64
                  else
                    idx = @kernel - 1 + back # back is negative; -1 -> last state column
                    idx >= 0 ? s[c, idx].to_f64 : 0.0
                  end
            acc += @weight[c, j].to_f64 * val
          end
          dst[t, c] = acc
        end
      end

      # Roll the state forward to the last kernel-1 positions of this call.
      if @kernel > 1
        keep = @kernel - 1
        new_s = SimpleMatrix.new(@channels, keep, 0.0)
        @channels.times do |c|
          keep.times do |i|
            # Fill right-aligned: the newest position lands in the last column.
            src = seq - keep + i
            new_s[c, i] = if src >= 0
                            x[src, c]
                          else
                            # The call was shorter than the window, so part of the old state
                            # survives. Without this a short chunk would silently lose history.
                            s[c, s.cols + src]
                          end
          end
        end
        s = new_s
      end

      {dst, s}
    end
  end
end
