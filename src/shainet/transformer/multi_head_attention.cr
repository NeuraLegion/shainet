module SHAInet
  class MultiHeadAttention
    getter num_heads, d_model, head_dim
    @head_dim : Int32
    @w_q : TensorMatrix
    @w_k : TensorMatrix
    @w_v : TensorMatrix
    @w_o : TensorMatrix
    @x_t : TensorMatrix?
    @out_t : TensorMatrix
    @x : SimpleMatrix?
    @q_heads : Array(SimpleMatrix)
    @k_heads : Array(SimpleMatrix)
    @v_heads : Array(SimpleMatrix)
    @attn : Array(SimpleMatrix)
    @out : SimpleMatrix

    getter w_q, w_k, w_v, w_o

    def initialize(@d_model : Int32, @num_heads : Int32)
      @head_dim = (@d_model // @num_heads)
      mat_klass = CUDA.available? ? CudaMatrix : SimpleMatrix
      @w_q = mat_klass.tensor(@d_model, @d_model).random_fill!
      @w_k = mat_klass.tensor(@d_model, @d_model).random_fill!
      @w_v = mat_klass.tensor(@d_model, @d_model).random_fill!
      @w_o = mat_klass.tensor(@d_model, @d_model).random_fill!
      @out_t = mat_klass.tensor(1, 1)
      @q_heads = [] of SimpleMatrix
      @k_heads = [] of SimpleMatrix
      @v_heads = [] of SimpleMatrix
      @attn = [] of SimpleMatrix
      @out = mat_klass.zeros(1, 1)
    end

    def forward(x : SimpleMatrix, mask : SimpleMatrix? = nil)
      @x = x
      @x_t = TensorMatrix.from_a(x.to_a)
      q_t = @x_t.not_nil! * @w_q
      k_t = @x_t.not_nil! * @w_k
      v_t = @x_t.not_nil! * @w_v
      q = x * @w_q.to_simple
      k = x * @w_k.to_simple
      v = x * @w_v.to_simple

      @q_heads = [] of SimpleMatrix
      @k_heads = [] of SimpleMatrix
      @v_heads = [] of SimpleMatrix
      @attn = [] of SimpleMatrix
      outputs = [] of SimpleMatrix
      q_heads_t = [] of TensorMatrix
      k_heads_t = [] of TensorMatrix
      v_heads_t = [] of TensorMatrix
      attn_t = [] of TensorMatrix
      outputs_t = [] of TensorMatrix

      @num_heads.times do |h|
        qs = q.slice_cols(h*@head_dim, @head_dim)
        ks = k.slice_cols(h*@head_dim, @head_dim)
        vs = v.slice_cols(h*@head_dim, @head_dim)
        qs_t = q_t.slice_cols(h*@head_dim, @head_dim)
        ks_t = k_t.slice_cols(h*@head_dim, @head_dim)
        vs_t = v_t.slice_cols(h*@head_dim, @head_dim)
        @q_heads << qs
        @k_heads << ks
        @v_heads << vs
        q_heads_t << qs_t
        k_heads_t << ks_t
        v_heads_t << vs_t

        scores = qs * ks.transpose * (1.0 / Math.sqrt(@head_dim.to_f))
        scores_t = qs_t * ks_t.transpose * (1.0 / Math.sqrt(@head_dim.to_f))
        if m = mask
          raise "mask size mismatch" unless m.rows == scores.rows && m.cols == scores.cols
          scores = scores + m
          scores_t = scores_t + TensorMatrix.from_a(m.to_a)
        end
        attn = softmax_rows(scores)
        attn_tensor = softmax_rows_tensor(scores_t)
        @attn << attn
        attn_t << attn_tensor
        outputs << (attn * vs)
        outputs_t << (attn_tensor * vs_t)
      end

      concat = SimpleMatrix.new(x.rows, @d_model)
      concat_t = TensorMatrix.new(x.rows, @d_model)
      @num_heads.times do |h|
        concat.set_cols!(h*@head_dim, outputs[h])
        concat_t.set_cols!(h*@head_dim, outputs_t[h])
      end

      @out_t = concat_t * @w_o
      @out = concat * @w_o.to_simple
      @out
    end

    def backward(d_out : SimpleMatrix)
      loss = Autograd::Tensor.new(0.0)
      d_out.rows.times do |i|
        d_out.cols.times do |j|
          loss = loss + @out_t[i, j] * Autograd::Tensor.new(d_out[i, j])
        end
      end
      loss.backward
      grad_matrix = @x_t.not_nil!.clone
      grad_matrix.rows.times do |i|
        grad_matrix.cols.times do |j|
          grad_matrix[i, j] = Autograd::Tensor.new(@x_t.not_nil![i, j].grad)
        end
      end
      grad_matrix.to_simple
    end

    def apply_gradients(lr : Float64)
      [@w_q, @w_k, @w_v, @w_o].each do |w|
        w.rows.times do |i|
          w.cols.times do |j|
            t = w[i, j]
            w[i, j] = Autograd::Tensor.new(t.data - lr * t.grad)
            t.grad = 0.0
          end
        end
      end
    end

    def zero_gradients
      [@w_q, @w_k, @w_v, @w_o].each &.zero_grads!
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

    private def softmax_rows_tensor(m : TensorMatrix)
      result = TensorMatrix.new(m.rows, m.cols)
      m.rows.times do |i|
        sum = Autograd::Tensor.new(0.0)
        m.cols.times { |j| sum = sum + Autograd::Tensor.new(Math.exp(m[i, j].data)) }
        m.cols.times do |j|
          result[i, j] = Autograd::Tensor.new(Math.exp(m[i, j].data)) / sum
        end
      end
      result
    end
  end
end
