module SHAInet
  module Autograd
    # GradMatrix is a matrix class that supports automatic differentiation.
    # Unlike TensorMatrix which wraps individual Tensor scalars, GradMatrix
    # operates at the matrix level for efficiency and proper matrix calculus.
    #
    # Example usage:
    #   x = GradMatrix.new([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
    #   w = GradMatrix.new([[0.5, 0.5], [0.5, 0.5]], requires_grad: true)
    #   y = x.matmul(w)
    #   loss = y.sum
    #   loss.backward
    #   puts x.grad  # Gradient of loss with respect to x
    #
    class GradMatrix
      property data : Array(Float64)
      property grad : Array(Float64)?
      property rows : Int32
      property cols : Int32
      property requires_grad : Bool
      property parents : Array(GradMatrix)
      property backward_fn : Proc(Nil)?
      property name : String?

      # Track if this is a leaf node (created by user, not from computation)
      getter is_leaf : Bool

      def initialize(@rows : Int32, @cols : Int32, init : Float64 = 0.0, @requires_grad : Bool = false, @name : String? = nil)
        @data = Array(Float64).new(@rows * @cols, init)
        @grad = @requires_grad ? Array(Float64).new(@rows * @cols, 0.0) : nil
        @parents = [] of GradMatrix
        @backward_fn = nil
        @is_leaf = true
      end

      def initialize(@rows : Int32, @cols : Int32, @data : Array(Float64), @requires_grad : Bool = false, @name : String? = nil)
        raise ArgumentError.new("Data size mismatch: expected #{@rows * @cols}, got #{@data.size}") unless @data.size == @rows * @cols
        @grad = @requires_grad ? Array(Float64).new(@rows * @cols, 0.0) : nil
        @parents = [] of GradMatrix
        @backward_fn = nil
        @is_leaf = true
      end

      # Create from 2D array
      def self.from_a(array : Array(Array(Float64)), requires_grad : Bool = false, name : String? = nil) : GradMatrix
        rows = array.size
        cols = array.first?.try(&.size) || 0
        data = Array(Float64).new(rows * cols)
        array.each do |row|
          raise ArgumentError.new("Inconsistent row sizes") unless row.size == cols
          row.each { |v| data << v }
        end
        new(rows, cols, data, requires_grad, name)
      end

      # Create zeros matrix
      def self.zeros(rows : Int32, cols : Int32, requires_grad : Bool = false) : GradMatrix
        new(rows, cols, 0.0, requires_grad)
      end

      # Create ones matrix
      def self.ones(rows : Int32, cols : Int32, requires_grad : Bool = false) : GradMatrix
        new(rows, cols, 1.0, requires_grad)
      end

      # Create random matrix
      def self.random(rows : Int32, cols : Int32, min : Float64 = -1.0, max : Float64 = 1.0, requires_grad : Bool = false) : GradMatrix
        data = Array(Float64).new(rows * cols) { Random.rand(min..max) }
        new(rows, cols, data, requires_grad)
      end

      # Create identity matrix
      def self.eye(size : Int32, requires_grad : Bool = false) : GradMatrix
        m = zeros(size, size, requires_grad)
        size.times { |i| m[i, i] = 1.0 }
        m
      end

      # Element access
      def [](row : Int32, col : Int32) : Float64
        @data[row * @cols + col]
      end

      def []=(row : Int32, col : Int32, value : Float64)
        @data[row * @cols + col] = value
      end

      # Get gradient value
      def grad_at(row : Int32, col : Int32) : Float64
        @grad.try { |g| g[row * @cols + col] } || 0.0
      end

      # Accumulate gradient
      def accumulate_grad!(grad_data : Array(Float64))
        if g = @grad
          grad_data.each_with_index { |v, i| g[i] += v }
        end
      end

      # Zero out gradients
      def zero_grad!
        @grad.try &.fill(0.0)
      end

      # Convert to 2D array
      def to_a : Array(Array(Float64))
        Array.new(@rows) do |i|
          Array.new(@cols) do |j|
            self[i, j]
          end
        end
      end

      # Clone the matrix (gradients are NOT copied, new matrix is a leaf)
      def clone : GradMatrix
        GradMatrix.new(@rows, @cols, @data.dup, @requires_grad)
      end

      # Detach from computation graph
      def detach : GradMatrix
        GradMatrix.new(@rows, @cols, @data.dup, false)
      end

      # Get size as tuple
      def shape : Tuple(Int32, Int32)
        {@rows, @cols}
      end

      # Total number of elements
      def size : Int32
        @rows * @cols
      end

      # -------------------------------------------------------------------
      # Matrix Operations with Autograd Support
      # -------------------------------------------------------------------

      # Matrix multiplication: C = A @ B
      # dA = dC @ B^T
      # dB = A^T @ dC
      def matmul(other : GradMatrix) : GradMatrix
        raise ArgumentError.new("Matmul dimension mismatch: #{@cols} != #{other.rows}") unless @cols == other.rows

        result_data = Array(Float64).new(@rows * other.cols, 0.0)
        @rows.times do |i|
          other.cols.times do |j|
            sum = 0.0
            @cols.times do |k|
              sum += self[i, k] * other[k, j]
            end
            result_data[i * other.cols + j] = sum
          end
        end

        result = GradMatrix.new(@rows, other.cols, result_data, @requires_grad || other.requires_grad)
        result.parents = [self, other]
        result.set_non_leaf!

        if result.requires_grad
          # Capture values for backward pass
          self_data = @data.dup
          self_rows = @rows
          self_cols = @cols
          other_data = other.data.dup
          other_rows = other.rows
          other_cols = other.cols

          result.backward_fn = -> do
            if out_grad = result.grad
              # dA = dC @ B^T
              if @requires_grad
                da = Array(Float64).new(self_rows * self_cols, 0.0)
                self_rows.times do |i|
                  self_cols.times do |j|
                    sum = 0.0
                    other_cols.times do |k|
                      sum += out_grad[i * other_cols + k] * other_data[j * other_cols + k]
                    end
                    da[i * self_cols + j] = sum
                  end
                end
                accumulate_grad!(da)
              end

              # dB = A^T @ dC
              if other.requires_grad
                db = Array(Float64).new(other_rows * other_cols, 0.0)
                other_rows.times do |i|
                  other_cols.times do |j|
                    sum = 0.0
                    self_rows.times do |k|
                      sum += self_data[k * self_cols + i] * out_grad[k * other_cols + j]
                    end
                    db[i * other_cols + j] = sum
                  end
                end
                other.accumulate_grad!(db)
              end
            end
          end
        end

        result
      end

      # Transpose: B = A^T
      # dA = dB^T
      def transpose : GradMatrix
        result_data = Array(Float64).new(@cols * @rows)
        @cols.times do |i|
          @rows.times do |j|
            result_data << self[j, i]
          end
        end

        result = GradMatrix.new(@cols, @rows, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          result_rows = result.rows
          result_cols = result.cols

          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(@rows * @cols, 0.0)
              result_rows.times do |i|
                result_cols.times do |j|
                  da[j * @cols + i] = out_grad[i * result_cols + j]
                end
              end
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # Element-wise addition: C = A + B
      def +(other : GradMatrix) : GradMatrix
        raise ArgumentError.new("Size mismatch for addition") unless @rows == other.rows && @cols == other.cols

        result_data = Array(Float64).new(@rows * @cols)
        @data.each_with_index { |v, i| result_data << v + other.data[i] }

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad || other.requires_grad)
        result.parents = [self, other]
        result.set_non_leaf!

        if result.requires_grad
          result.backward_fn = -> do
            if out_grad = result.grad
              accumulate_grad!(out_grad) if @requires_grad
              other.accumulate_grad!(out_grad) if other.requires_grad
            end
          end
        end

        result
      end

      # Element-wise subtraction: C = A - B
      def -(other : GradMatrix) : GradMatrix
        raise ArgumentError.new("Size mismatch for subtraction") unless @rows == other.rows && @cols == other.cols

        result_data = Array(Float64).new(@rows * @cols)
        @data.each_with_index { |v, i| result_data << v - other.data[i] }

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad || other.requires_grad)
        result.parents = [self, other]
        result.set_non_leaf!

        if result.requires_grad
          result.backward_fn = -> do
            if out_grad = result.grad
              accumulate_grad!(out_grad) if @requires_grad
              if other.requires_grad
                neg_grad = out_grad.map { |v| -v }
                other.accumulate_grad!(neg_grad)
              end
            end
          end
        end

        result
      end

      # Element-wise multiplication (Hadamard product): C = A ⊙ B
      def hadamard(other : GradMatrix) : GradMatrix
        raise ArgumentError.new("Size mismatch for Hadamard product") unless @rows == other.rows && @cols == other.cols

        result_data = Array(Float64).new(@rows * @cols)
        @data.each_with_index { |v, i| result_data << v * other.data[i] }

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad || other.requires_grad)
        result.parents = [self, other]
        result.set_non_leaf!

        if result.requires_grad
          self_data = @data.dup
          other_data = other.data.dup

          result.backward_fn = -> do
            if out_grad = result.grad
              if @requires_grad
                da = Array(Float64).new(@rows * @cols)
                out_grad.each_with_index { |g, i| da << g * other_data[i] }
                accumulate_grad!(da)
              end
              if other.requires_grad
                db = Array(Float64).new(@rows * @cols)
                out_grad.each_with_index { |g, i| db << g * self_data[i] }
                other.accumulate_grad!(db)
              end
            end
          end
        end

        result
      end

      # Scalar multiplication: C = A * scalar
      def *(scalar : Float64) : GradMatrix
        result_data = @data.map { |v| v * scalar }

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          result.backward_fn = -> do
            if out_grad = result.grad
              da = out_grad.map { |g| g * scalar }
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # Scalar division: C = A / scalar
      def /(scalar : Float64) : GradMatrix
        self * (1.0 / scalar)
      end

      # Negation: B = -A
      def - : GradMatrix
        self * (-1.0)
      end

      # Sum all elements: scalar = sum(A)
      def sum : GradMatrix
        total = @data.sum
        result = GradMatrix.new(1, 1, [total], @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          n = @rows * @cols
          result.backward_fn = -> do
            if out_grad = result.grad
              # Gradient flows to all elements equally
              da = Array(Float64).new(n, out_grad[0])
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # Mean of all elements: scalar = mean(A)
      def mean : GradMatrix
        avg = @data.sum / (@rows * @cols)
        result = GradMatrix.new(1, 1, [avg], @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          n = @rows * @cols
          result.backward_fn = -> do
            if out_grad = result.grad
              scale = out_grad[0] / n
              da = Array(Float64).new(n, scale)
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # Sum along rows: result[i] = sum of row i
      def sum_rows : GradMatrix
        result_data = Array(Float64).new(@rows)
        @rows.times do |i|
          sum = 0.0
          @cols.times { |j| sum += self[i, j] }
          result_data << sum
        end

        result = GradMatrix.new(@rows, 1, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          cols = @cols
          rows = @rows
          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(rows * cols)
              rows.times do |i|
                g = out_grad[i]
                cols.times { da << g }
              end
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # Sum along columns: result[j] = sum of column j
      def sum_cols : GradMatrix
        result_data = Array(Float64).new(@cols, 0.0)
        @rows.times do |i|
          @cols.times { |j| result_data[j] += self[i, j] }
        end

        result = GradMatrix.new(1, @cols, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          cols = @cols
          rows = @rows
          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(rows * cols)
              rows.times do |_i|
                cols.times { |j| da << out_grad[j] }
              end
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # Power: C = A^p (element-wise)
      def pow(p : Float64) : GradMatrix
        result_data = @data.map { |v| v ** p }

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          self_data = @data.dup
          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(@rows * @cols)
              out_grad.each_with_index do |g, i|
                da << g * p * (self_data[i] ** (p - 1))
              end
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # Square root: C = sqrt(A) (element-wise)
      def sqrt : GradMatrix
        result_data = @data.map { |v| Math.sqrt(v) }

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          result_data_copy = result_data.dup
          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(@rows * @cols)
              out_grad.each_with_index do |g, i|
                da << g / (2.0 * result_data_copy[i])
              end
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # Exponential: C = exp(A) (element-wise)
      def exp : GradMatrix
        result_data = @data.map { |v| Math.exp(v) }

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          result_data_copy = result_data.dup
          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(@rows * @cols)
              out_grad.each_with_index do |g, i|
                da << g * result_data_copy[i]
              end
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # Natural log: C = log(A) (element-wise)
      def log : GradMatrix
        result_data = @data.map { |v| Math.log(v) }

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          self_data = @data.dup
          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(@rows * @cols)
              out_grad.each_with_index do |g, i|
                da << g / self_data[i]
              end
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # -------------------------------------------------------------------
      # Activation Functions
      # -------------------------------------------------------------------

      # ReLU: C = max(0, A)
      def relu : GradMatrix
        result_data = @data.map { |v| v > 0 ? v : 0.0 }

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          self_data = @data.dup
          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(@rows * @cols)
              out_grad.each_with_index do |g, i|
                da << (self_data[i] > 0 ? g : 0.0)
              end
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # Leaky ReLU: C = max(alpha * A, A)
      def leaky_relu(alpha : Float64 = 0.01) : GradMatrix
        result_data = @data.map { |v| v > 0 ? v : alpha * v }

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          self_data = @data.dup
          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(@rows * @cols)
              out_grad.each_with_index do |g, i|
                da << (self_data[i] > 0 ? g : g * alpha)
              end
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # Sigmoid: C = 1 / (1 + exp(-A))
      def sigmoid : GradMatrix
        result_data = @data.map { |v| 1.0 / (1.0 + Math.exp(-v)) }

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          result_data_copy = result_data.dup
          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(@rows * @cols)
              out_grad.each_with_index do |g, i|
                s = result_data_copy[i]
                da << g * s * (1.0 - s)
              end
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # Tanh: C = tanh(A)
      def tanh : GradMatrix
        result_data = @data.map { |v| Math.tanh(v) }

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          result_data_copy = result_data.dup
          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(@rows * @cols)
              out_grad.each_with_index do |g, i|
                t = result_data_copy[i]
                da << g * (1.0 - t * t)
              end
              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # Softmax along rows (each row sums to 1)
      def softmax : GradMatrix
        result_data = Array(Float64).new(@rows * @cols)

        @rows.times do |i|
          # Find max for numerical stability
          row_max = -Float64::INFINITY
          @cols.times { |j| row_max = Math.max(row_max, self[i, j]) }

          # Compute exp and sum
          row_exp = Array(Float64).new(@cols)
          row_sum = 0.0
          @cols.times do |j|
            e = Math.exp(self[i, j] - row_max)
            row_exp << e
            row_sum += e
          end

          # Normalize
          row_exp.each { |e| result_data << e / row_sum }
        end

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          result_data_copy = result_data.dup
          rows = @rows
          cols = @cols

          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(rows * cols, 0.0)

              rows.times do |i|
                # For each row, compute Jacobian * out_grad
                # softmax Jacobian: diag(s) - s * s^T
                row_start = i * cols

                cols.times do |j|
                  sum = 0.0
                  cols.times do |k|
                    s_j = result_data_copy[row_start + j]
                    s_k = result_data_copy[row_start + k]
                    jacobian_jk = if j == k
                                    s_j * (1.0 - s_j)
                                  else
                                    -s_j * s_k
                                  end
                    sum += jacobian_jk * out_grad[row_start + k]
                  end
                  da[row_start + j] = sum
                end
              end

              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # GELU activation (Gaussian Error Linear Unit)
      def gelu : GradMatrix
        # GELU(x) ≈ x * Φ(x) where Φ is the CDF of standard normal
        # Using the tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = Math.sqrt(2.0 / Math::PI)

        result_data = @data.map do |x|
          0.5 * x * (1.0 + Math.tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)))
        end

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          self_data = @data.dup

          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(@rows * @cols)

              out_grad.each_with_index do |g, i|
                x = self_data[i]
                # Derivative of GELU (tanh approximation)
                inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
                tanh_inner = Math.tanh(inner)
                sech2 = 1.0 - tanh_inner * tanh_inner
                d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x)

                d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
                da << g * d_gelu
              end

              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # -------------------------------------------------------------------
      # Loss Functions
      # -------------------------------------------------------------------

      # Mean Squared Error: MSE = mean((pred - target)^2)
      def mse_loss(target : GradMatrix) : GradMatrix
        raise ArgumentError.new("Shape mismatch") unless @rows == target.rows && @cols == target.cols
        diff = self - target
        squared = diff.hadamard(diff)
        squared.mean
      end

      # Cross-entropy loss (for softmax outputs and one-hot targets)
      # CE = -sum(target * log(pred + eps))
      def cross_entropy_loss(target : GradMatrix, eps : Float64 = 1e-10) : GradMatrix
        raise ArgumentError.new("Shape mismatch") unless @rows == target.rows && @cols == target.cols

        # Add epsilon for numerical stability and take log
        log_pred_data = @data.map { |v| Math.log(v + eps) }
        log_pred = GradMatrix.new(@rows, @cols, log_pred_data, @requires_grad)
        log_pred.parents = [self]
        log_pred.set_non_leaf!

        if @requires_grad
          self_data = @data.dup
          log_pred.backward_fn = -> do
            if out_grad = log_pred.grad
              da = Array(Float64).new(@rows * @cols)
              out_grad.each_with_index do |g, i|
                da << g / (self_data[i] + eps)
              end
              accumulate_grad!(da)
            end
          end
        end

        # Compute -sum(target * log_pred)
        product = target.hadamard(log_pred)
        neg_sum = product.sum * (-1.0)
        neg_sum
      end

      # Binary cross-entropy: BCE = -mean(target * log(pred) + (1-target) * log(1-pred))
      def binary_cross_entropy_loss(target : GradMatrix, eps : Float64 = 1e-10) : GradMatrix
        raise ArgumentError.new("Shape mismatch") unless @rows == target.rows && @cols == target.cols

        n = (@rows * @cols).to_f64
        loss_sum = 0.0

        @data.each_with_index do |pred, i|
          t = target.data[i]
          pred_clipped = Math.max(eps, Math.min(1.0 - eps, pred))
          loss_sum -= t * Math.log(pred_clipped) + (1.0 - t) * Math.log(1.0 - pred_clipped)
        end

        result = GradMatrix.new(1, 1, [loss_sum / n], @requires_grad)
        result.parents = [self]
        result.set_non_leaf!

        if result.requires_grad
          self_data = @data.dup
          target_data = target.data.dup

          result.backward_fn = -> do
            if out_grad = result.grad
              da = Array(Float64).new(@rows * @cols)
              scale = out_grad[0] / n

              self_data.each_with_index do |pred, i|
                t = target_data[i]
                pred_clipped = Math.max(eps, Math.min(1.0 - eps, pred))
                # d(BCE)/d(pred) = -(t/pred - (1-t)/(1-pred))
                grad = -scale * (t / pred_clipped - (1.0 - t) / (1.0 - pred_clipped))
                da << grad
              end

              accumulate_grad!(da)
            end
          end
        end

        result
      end

      # -------------------------------------------------------------------
      # Backward Pass
      # -------------------------------------------------------------------

      # Perform backward pass starting from this tensor
      def backward(initial_grad : Array(Float64)? = nil)
        # Initialize gradient for this tensor
        if g = @grad
          if initial_grad
            raise ArgumentError.new("Initial grad size mismatch") unless initial_grad.size == g.size
            initial_grad.each_with_index { |v, i| g[i] = v }
          else
            # Default: assume this is a scalar loss, gradient = 1
            g.fill(1.0)
          end
        end

        # Build topological order
        topo = build_topo

        # Execute backward functions in reverse topological order
        topo.reverse_each do |node|
          node.backward_fn.try &.call
        end
      end

      protected def set_non_leaf!
        @is_leaf = false
      end

      private def build_topo : Array(GradMatrix)
        visited = Set(GradMatrix).new
        topo = [] of GradMatrix
        dfs(self, visited, topo)
        topo
      end

      private def dfs(node : GradMatrix, visited : Set(GradMatrix), topo : Array(GradMatrix))
        return if visited.includes?(node)
        visited.add(node)
        node.parents.each { |p| dfs(p, visited, topo) }
        topo << node
      end

      # -------------------------------------------------------------------
      # Broadcasting Operations
      # -------------------------------------------------------------------

      # Add a row vector to each row: C[i, j] = A[i, j] + B[0, j]
      def add_row_broadcast(row_vec : GradMatrix) : GradMatrix
        raise ArgumentError.new("Row vector cols mismatch") unless row_vec.rows == 1 && row_vec.cols == @cols

        result_data = Array(Float64).new(@rows * @cols)
        @rows.times do |i|
          @cols.times do |j|
            result_data << self[i, j] + row_vec[0, j]
          end
        end

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad || row_vec.requires_grad)
        result.parents = [self, row_vec]
        result.set_non_leaf!

        if result.requires_grad
          rows = @rows

          result.backward_fn = -> do
            if out_grad = result.grad
              accumulate_grad!(out_grad) if @requires_grad

              if row_vec.requires_grad
                # Sum gradients along rows
                db = Array(Float64).new(@cols, 0.0)
                rows.times do |i|
                  @cols.times { |j| db[j] += out_grad[i * @cols + j] }
                end
                row_vec.accumulate_grad!(db)
              end
            end
          end
        end

        result
      end

      # Multiply each row by a row vector element-wise: C[i, j] = A[i, j] * B[0, j]
      def mul_row_broadcast(row_vec : GradMatrix) : GradMatrix
        raise ArgumentError.new("Row vector cols mismatch") unless row_vec.rows == 1 && row_vec.cols == @cols

        result_data = Array(Float64).new(@rows * @cols)
        @rows.times do |i|
          @cols.times do |j|
            result_data << self[i, j] * row_vec[0, j]
          end
        end

        result = GradMatrix.new(@rows, @cols, result_data, @requires_grad || row_vec.requires_grad)
        result.parents = [self, row_vec]
        result.set_non_leaf!

        if result.requires_grad
          self_data = @data.dup
          row_vec_data = row_vec.data.dup
          rows = @rows
          cols = @cols

          result.backward_fn = -> do
            if out_grad = result.grad
              if @requires_grad
                da = Array(Float64).new(rows * cols)
                rows.times do |i|
                  cols.times { |j| da << out_grad[i * cols + j] * row_vec_data[j] }
                end
                accumulate_grad!(da)
              end

              if row_vec.requires_grad
                db = Array(Float64).new(cols, 0.0)
                rows.times do |i|
                  cols.times { |j| db[j] += out_grad[i * cols + j] * self_data[i * cols + j] }
                end
                row_vec.accumulate_grad!(db)
              end
            end
          end
        end

        result
      end

      # -------------------------------------------------------------------
      # Utility Methods
      # -------------------------------------------------------------------

      def to_s(io : IO)
        io << "GradMatrix(#{@rows}x#{@cols}"
        io << ", requires_grad=true" if @requires_grad
        io << ", name=#{@name}" if @name
        io << ")"
      end

      def inspect(io : IO)
        io << "GradMatrix(\n"
        @rows.times do |i|
          io << "  ["
          @cols.times do |j|
            io << ", " if j > 0
            io << sprintf("%.4f", self[i, j])
          end
          io << "]\n"
        end
        io << ")"
      end

      # Get gradient as 2D array
      def grad_to_a : Array(Array(Float64))?
        return nil unless @grad
        Array.new(@rows) do |i|
          Array.new(@cols) do |j|
            grad_at(i, j)
          end
        end
      end
    end
  end
end
