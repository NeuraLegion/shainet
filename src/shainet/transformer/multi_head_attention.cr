module SHAInet
  class MultiHeadAttention
    getter num_heads, d_model, head_dim
    @head_dim : Int32
    @w_q : SimpleMatrix
    @w_k : SimpleMatrix
    @w_v : SimpleMatrix
    @w_o : SimpleMatrix
    @grads_w_q : SimpleMatrix
    @grads_w_k : SimpleMatrix
    @grads_w_v : SimpleMatrix
    @grads_w_o : SimpleMatrix
    @x : SimpleMatrix?
    @q_heads : Array(SimpleMatrix)
    @k_heads : Array(SimpleMatrix)
    @v_heads : Array(SimpleMatrix)
    @attn : Array(SimpleMatrix)
    @out : SimpleMatrix

    getter w_q, w_k, w_v, w_o
    property grads_w_q : SimpleMatrix
    property grads_w_k : SimpleMatrix
    property grads_w_v : SimpleMatrix
    property grads_w_o : SimpleMatrix

    def initialize(@d_model : Int32, @num_heads : Int32)
      @head_dim = (@d_model // @num_heads)
      mat_klass = CUDA.available? ? CudaMatrix : SimpleMatrix
      @w_q = mat_klass.new(@d_model, @d_model).random_fill!
      @w_k = mat_klass.new(@d_model, @d_model).random_fill!
      @w_v = mat_klass.new(@d_model, @d_model).random_fill!
      @w_o = mat_klass.new(@d_model, @d_model).random_fill!
      @grads_w_q = mat_klass.zeros(@d_model, @d_model)
      @grads_w_k = mat_klass.zeros(@d_model, @d_model)
      @grads_w_v = mat_klass.zeros(@d_model, @d_model)
      @grads_w_o = mat_klass.zeros(@d_model, @d_model)
      @q_heads = [] of SimpleMatrix
      @k_heads = [] of SimpleMatrix
      @v_heads = [] of SimpleMatrix
      @attn = [] of SimpleMatrix
      @out = mat_klass.zeros(1, 1)
    end

    def forward(x : SimpleMatrix, mask : SimpleMatrix? = nil)
      @x = x
      q = x * @w_q
      k = x * @w_k
      v = x * @w_v

      @q_heads = [] of SimpleMatrix
      @k_heads = [] of SimpleMatrix
      @v_heads = [] of SimpleMatrix
      @attn = [] of SimpleMatrix
      outputs = [] of SimpleMatrix

      @num_heads.times do |h|
        qs = q.slice_cols(h*@head_dim, @head_dim)
        ks = k.slice_cols(h*@head_dim, @head_dim)
        vs = v.slice_cols(h*@head_dim, @head_dim)
        @q_heads << qs
        @k_heads << ks
        @v_heads << vs

        scores = qs * ks.transpose * (1.0 / Math.sqrt(@head_dim.to_f))
        if m = mask
          raise "mask size mismatch" unless m.rows == scores.rows && m.cols == scores.cols
          scores = scores + m
        end
        attn = softmax_rows(scores)
        @attn << attn
        outputs << (attn * vs)
      end

      concat = SimpleMatrix.new(x.rows, @d_model)
      @num_heads.times do |h|
        concat.set_cols!(h*@head_dim, outputs[h])
      end

      @out = concat * @w_o
      @out
    end

    def backward(d_out : SimpleMatrix)
      # Gradients for output projection
      mat_klass = @w_q.class
      @grads_w_o = (@q_heads.size == 0 ? mat_klass.zeros(@d_model, @d_model) : @grads_w_o)
      @grads_w_o = @grads_w_o + ((@out.clone.transpose * d_out))
      d_concat = d_out * @w_o.transpose

      d_q_total = mat_klass.zeros(@x.not_nil!.rows, @d_model)
      d_k_total = mat_klass.zeros(@x.not_nil!.rows, @d_model)
      d_v_total = mat_klass.zeros(@x.not_nil!.rows, @d_model)

      @num_heads.times do |h|
        d_head = d_concat.slice_cols(h*@head_dim, @head_dim)
        attn = @attn[h]
        vs = @v_heads[h]
        qs = @q_heads[h]
        ks = @k_heads[h]

        d_attn = d_head * vs.transpose
        d_vs = attn.transpose * d_head

        # softmax gradient
        d_scores = mat_klass.zeros(attn.rows, attn.cols)
        attn.rows.times do |i|
          sum = 0.0
          attn.cols.times do |j|
            sum += attn[i, j] * d_attn[i, j]
          end
          attn.cols.times do |j|
            d_scores[i, j] = attn[i, j]*(d_attn[i, j] - sum)
          end
        end
        d_qs = d_scores * ks
        d_ks = d_scores.transpose * qs

        d_q_total.set_cols!(h*@head_dim, d_qs)
        d_k_total.set_cols!(h*@head_dim, d_ks)
        d_v_total.set_cols!(h*@head_dim, d_vs)
      end

      @grads_w_q = @grads_w_q + (@x.not_nil!.transpose * d_q_total)
      @grads_w_k = @grads_w_k + (@x.not_nil!.transpose * d_k_total)
      @grads_w_v = @grads_w_v + (@x.not_nil!.transpose * d_v_total)

      d_input = d_q_total * @w_q.transpose + d_k_total * @w_k.transpose + d_v_total * @w_v.transpose

      d_input
    end

    def apply_gradients(lr : Float64)
      mat_klass = @w_q.class
      @w_q = @w_q - @grads_w_q * lr
      @w_k = @w_k - @grads_w_k * lr
      @w_v = @w_v - @grads_w_v * lr
      @w_o = @w_o - @grads_w_o * lr
      @grads_w_q = mat_klass.zeros(@d_model, @d_model)
      @grads_w_k = mat_klass.zeros(@d_model, @d_model)
      @grads_w_v = mat_klass.zeros(@d_model, @d_model)
      @grads_w_o = mat_klass.zeros(@d_model, @d_model)
    end

    def zero_gradients
      mat_klass = @w_q.class
      @grads_w_q = mat_klass.zeros(@d_model, @d_model)
      @grads_w_k = mat_klass.zeros(@d_model, @d_model)
      @grads_w_v = mat_klass.zeros(@d_model, @d_model)
      @grads_w_o = mat_klass.zeros(@d_model, @d_model)
    end

    private def softmax_rows(m : SimpleMatrix)
      result = SimpleMatrix.new(m.rows, m.cols)
      m.rows.times do |i|
        sum = 0.0
        m.cols.times { |j| sum += Math.exp(m[i, j]) }
        m.cols.times { |j| result[i, j] = Math.exp(m[i, j]) / sum }
      end
      result
    end
  end
end
