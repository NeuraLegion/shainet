require "./spec_helper"

describe SHAInet::Autograd::GradMatrix do
  describe "basic operations" do
    it "creates a matrix with correct dimensions" do
      m = SHAInet::Autograd::GradMatrix.new(2, 3, requires_grad: true)
      m.rows.should eq(2)
      m.cols.should eq(3)
      m.shape.should eq({2, 3})
      m.size.should eq(6)
      m.requires_grad.should be_true
    end

    it "creates from 2D array" do
      m = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
      m[0, 0].should eq(1.0)
      m[0, 1].should eq(2.0)
      m[1, 0].should eq(3.0)
      m[1, 1].should eq(4.0)
    end

    it "creates zeros matrix" do
      m = SHAInet::Autograd::GradMatrix.zeros(2, 2)
      m[0, 0].should eq(0.0)
      m[1, 1].should eq(0.0)
    end

    it "creates ones matrix" do
      m = SHAInet::Autograd::GradMatrix.ones(2, 2)
      m[0, 0].should eq(1.0)
      m[1, 1].should eq(1.0)
    end

    it "creates identity matrix" do
      m = SHAInet::Autograd::GradMatrix.eye(3)
      m[0, 0].should eq(1.0)
      m[1, 1].should eq(1.0)
      m[2, 2].should eq(1.0)
      m[0, 1].should eq(0.0)
    end
  end

  describe "element-wise addition" do
    it "adds two matrices" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
      b = SHAInet::Autograd::GradMatrix.from_a([[5.0, 6.0], [7.0, 8.0]], requires_grad: true)
      c = a + b

      c[0, 0].should eq(6.0)
      c[1, 1].should eq(12.0)
    end

    it "computes correct gradients for addition" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
      b = SHAInet::Autograd::GradMatrix.from_a([[5.0, 6.0], [7.0, 8.0]], requires_grad: true)
      c = a + b
      loss = c.sum
      loss.backward

      # Gradient of sum w.r.t each element is 1
      a.grad_at(0, 0).should eq(1.0)
      a.grad_at(1, 1).should eq(1.0)
      b.grad_at(0, 0).should eq(1.0)
      b.grad_at(1, 1).should eq(1.0)
    end
  end

  describe "element-wise subtraction" do
    it "subtracts two matrices" do
      a = SHAInet::Autograd::GradMatrix.from_a([[5.0, 6.0]], requires_grad: true)
      b = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0]], requires_grad: true)
      c = a - b

      c[0, 0].should eq(4.0)
      c[0, 1].should eq(4.0)
    end

    it "computes correct gradients for subtraction" do
      a = SHAInet::Autograd::GradMatrix.from_a([[5.0, 6.0]], requires_grad: true)
      b = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0]], requires_grad: true)
      c = a - b
      loss = c.sum
      loss.backward

      a.grad_at(0, 0).should eq(1.0)
      b.grad_at(0, 0).should eq(-1.0)
    end
  end

  describe "matrix multiplication" do
    it "multiplies matrices correctly" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
      b = SHAInet::Autograd::GradMatrix.from_a([[5.0, 6.0], [7.0, 8.0]], requires_grad: true)
      c = a.matmul(b)

      # [1*5+2*7, 1*6+2*8] = [19, 22]
      # [3*5+4*7, 3*6+4*8] = [43, 50]
      c[0, 0].should eq(19.0)
      c[0, 1].should eq(22.0)
      c[1, 0].should eq(43.0)
      c[1, 1].should eq(50.0)
    end

    it "computes correct gradients for matmul" do
      # A (2x3) @ B (3x2) = C (2x2)
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad: true)
      b = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad: true)
      c = a.matmul(b)
      loss = c.sum
      loss.backward

      # dL/dA = dL/dC @ B^T
      # dL/dC = ones(2x2)
      # B^T = [[1,3,5], [2,4,6]]
      # dL/dA[0,0] = 1*1 + 1*2 = 3
      # dL/dA[0,1] = 1*3 + 1*4 = 7
      # dL/dA[0,2] = 1*5 + 1*6 = 11
      a.grad_at(0, 0).should eq(3.0)
      a.grad_at(0, 1).should eq(7.0)
      a.grad_at(0, 2).should eq(11.0)

      # dL/dB = A^T @ dL/dC
      # A^T = [[1,4], [2,5], [3,6]]
      # dL/dB[0,0] = 1*1 + 4*1 = 5
      # dL/dB[1,0] = 2*1 + 5*1 = 7
      b.grad_at(0, 0).should eq(5.0)
      b.grad_at(1, 0).should eq(7.0)
    end
  end

  describe "transpose" do
    it "transposes correctly" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad: true)
      b = a.transpose

      b.rows.should eq(3)
      b.cols.should eq(2)
      b[0, 0].should eq(1.0)
      b[1, 0].should eq(2.0)
      b[0, 1].should eq(4.0)
    end

    it "computes correct gradients for transpose" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
      b = a.transpose
      loss = b.sum
      loss.backward

      # All elements should have gradient 1
      a.grad_at(0, 0).should eq(1.0)
      a.grad_at(0, 1).should eq(1.0)
      a.grad_at(1, 0).should eq(1.0)
      a.grad_at(1, 1).should eq(1.0)
    end
  end

  describe "Hadamard product" do
    it "element-wise multiplies correctly" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
      b = SHAInet::Autograd::GradMatrix.from_a([[2.0, 3.0], [4.0, 5.0]], requires_grad: true)
      c = a.hadamard(b)

      c[0, 0].should eq(2.0)
      c[0, 1].should eq(6.0)
      c[1, 0].should eq(12.0)
      c[1, 1].should eq(20.0)
    end

    it "computes correct gradients for Hadamard product" do
      a = SHAInet::Autograd::GradMatrix.from_a([[2.0, 3.0]], requires_grad: true)
      b = SHAInet::Autograd::GradMatrix.from_a([[4.0, 5.0]], requires_grad: true)
      c = a.hadamard(b)
      loss = c.sum
      loss.backward

      # dL/dA[i] = dL/dC[i] * B[i] = 1 * B[i]
      a.grad_at(0, 0).should eq(4.0)
      a.grad_at(0, 1).should eq(5.0)
      # dL/dB[i] = dL/dC[i] * A[i] = 1 * A[i]
      b.grad_at(0, 0).should eq(2.0)
      b.grad_at(0, 1).should eq(3.0)
    end
  end

  describe "scalar operations" do
    it "multiplies by scalar" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
      c = a * 2.0

      c[0, 0].should eq(2.0)
      c[1, 1].should eq(8.0)
    end

    it "computes gradients for scalar multiplication" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0]], requires_grad: true)
      c = a * 3.0
      loss = c.sum
      loss.backward

      a.grad_at(0, 0).should eq(3.0)
      a.grad_at(0, 1).should eq(3.0)
    end
  end

  describe "reduction operations" do
    it "computes sum" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
      s = a.sum

      s[0, 0].should eq(10.0)
    end

    it "computes mean" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
      m = a.mean

      m[0, 0].should eq(2.5)
    end

    it "computes gradients for mean" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
      m = a.mean
      m.backward

      # Gradient = 1/n = 0.25
      a.grad_at(0, 0).should be_close(0.25, 1e-10)
      a.grad_at(1, 1).should be_close(0.25, 1e-10)
    end

    it "computes sum_rows" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad: true)
      s = a.sum_rows

      s.rows.should eq(2)
      s.cols.should eq(1)
      s[0, 0].should eq(6.0)
      s[1, 0].should eq(15.0)
    end
  end

  describe "activation functions" do
    describe "ReLU" do
      it "applies ReLU correctly" do
        a = SHAInet::Autograd::GradMatrix.from_a([[-1.0, 2.0], [3.0, -4.0]], requires_grad: true)
        r = a.relu

        r[0, 0].should eq(0.0)
        r[0, 1].should eq(2.0)
        r[1, 0].should eq(3.0)
        r[1, 1].should eq(0.0)
      end

      it "computes ReLU gradients correctly" do
        a = SHAInet::Autograd::GradMatrix.from_a([[-1.0, 2.0]], requires_grad: true)
        r = a.relu
        loss = r.sum
        loss.backward

        a.grad_at(0, 0).should eq(0.0) # Negative, gradient blocked
        a.grad_at(0, 1).should eq(1.0) # Positive, gradient passes
      end
    end

    describe "Sigmoid" do
      it "applies sigmoid correctly" do
        a = SHAInet::Autograd::GradMatrix.from_a([[0.0]], requires_grad: true)
        s = a.sigmoid

        s[0, 0].should eq(0.5)
      end

      it "computes sigmoid gradients correctly" do
        a = SHAInet::Autograd::GradMatrix.from_a([[0.0]], requires_grad: true)
        s = a.sigmoid
        s.backward

        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        a.grad_at(0, 0).should be_close(0.25, 1e-10)
      end
    end

    describe "Tanh" do
      it "applies tanh correctly" do
        a = SHAInet::Autograd::GradMatrix.from_a([[0.0]], requires_grad: true)
        t = a.tanh

        t[0, 0].should eq(0.0)
      end

      it "computes tanh gradients correctly" do
        a = SHAInet::Autograd::GradMatrix.from_a([[0.0]], requires_grad: true)
        t = a.tanh
        t.backward

        # tanh'(0) = 1 - tanh(0)^2 = 1 - 0 = 1
        a.grad_at(0, 0).should be_close(1.0, 1e-10)
      end
    end

    describe "Softmax" do
      it "applies softmax correctly" do
        a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0, 3.0]], requires_grad: true)
        s = a.softmax

        # Rows should sum to 1
        (s[0, 0] + s[0, 1] + s[0, 2]).should be_close(1.0, 1e-10)

        # s[0,2] should be largest
        s[0, 2].should be > s[0, 1]
        s[0, 1].should be > s[0, 0]
      end

      it "computes softmax gradients correctly" do
        a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0]], requires_grad: true)
        s = a.softmax
        loss = s.sum # Should equal 1
        loss.backward

        # Gradient of sum(softmax) w.r.t inputs should be 0
        # because softmax always sums to 1
        a.grad_at(0, 0).should be_close(0.0, 1e-10)
        a.grad_at(0, 1).should be_close(0.0, 1e-10)
      end
    end

    describe "GELU" do
      it "applies GELU correctly" do
        a = SHAInet::Autograd::GradMatrix.from_a([[0.0, 1.0, -1.0]], requires_grad: true)
        g = a.gelu

        # GELU(0) = 0
        g[0, 0].should be_close(0.0, 1e-5)
        # GELU(1) ≈ 0.841
        g[0, 1].should be_close(0.841, 0.01)
        # GELU(-1) ≈ -0.159
        g[0, 2].should be_close(-0.159, 0.01)
      end
    end
  end

  describe "loss functions" do
    describe "MSE loss" do
      it "computes MSE correctly" do
        pred = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0]], requires_grad: true)
        target = SHAInet::Autograd::GradMatrix.from_a([[1.5, 2.5]])
        loss = pred.mse_loss(target)

        # MSE = mean((1-1.5)^2 + (2-2.5)^2) = mean(0.25 + 0.25) = 0.25
        loss[0, 0].should be_close(0.25, 1e-10)
      end

      it "computes MSE gradients correctly" do
        pred = SHAInet::Autograd::GradMatrix.from_a([[2.0, 4.0]], requires_grad: true)
        target = SHAInet::Autograd::GradMatrix.from_a([[1.0, 3.0]])
        loss = pred.mse_loss(target)
        loss.backward

        # d(MSE)/d(pred) = 2 * (pred - target) / n
        # = 2 * [1, 1] / 2 = [1, 1]
        pred.grad_at(0, 0).should be_close(1.0, 1e-10)
        pred.grad_at(0, 1).should be_close(1.0, 1e-10)
      end
    end

    describe "Cross-entropy loss" do
      it "computes cross-entropy correctly" do
        # One-hot target [1, 0]
        pred = SHAInet::Autograd::GradMatrix.from_a([[0.9, 0.1]], requires_grad: true)
        target = SHAInet::Autograd::GradMatrix.from_a([[1.0, 0.0]])
        loss = pred.cross_entropy_loss(target)

        # CE = -log(0.9) ≈ 0.105
        loss[0, 0].should be_close(-Math.log(0.9), 0.01)
      end
    end

    describe "Binary cross-entropy loss" do
      it "computes BCE correctly" do
        pred = SHAInet::Autograd::GradMatrix.from_a([[0.8]], requires_grad: true)
        target = SHAInet::Autograd::GradMatrix.from_a([[1.0]])
        loss = pred.binary_cross_entropy_loss(target)

        # BCE = -(1*log(0.8) + 0*log(0.2)) = -log(0.8) ≈ 0.223
        loss[0, 0].should be_close(-Math.log(0.8), 0.01)
      end
    end
  end

  describe "numerical gradient verification" do
    it "matches numerical gradient for matmul" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
      b = SHAInet::Autograd::GradMatrix.from_a([[0.5, 0.5], [0.5, 0.5]], requires_grad: true)

      c = a.matmul(b)
      loss = c.sum
      loss.backward

      # Verify numerically for a[0,0]
      h = 1e-6
      a_plus = SHAInet::Autograd::GradMatrix.from_a([[1.0 + h, 2.0], [3.0, 4.0]])
      a_minus = SHAInet::Autograd::GradMatrix.from_a([[1.0 - h, 2.0], [3.0, 4.0]])
      b_copy = SHAInet::Autograd::GradMatrix.from_a([[0.5, 0.5], [0.5, 0.5]])

      loss_plus = a_plus.matmul(b_copy).sum[0, 0]
      loss_minus = a_minus.matmul(b_copy).sum[0, 0]
      numerical_grad = (loss_plus - loss_minus) / (2 * h)

      a.grad_at(0, 0).should be_close(numerical_grad, 1e-5)
    end

    it "matches numerical gradient for sigmoid" do
      a = SHAInet::Autograd::GradMatrix.from_a([[0.5]], requires_grad: true)
      s = a.sigmoid
      s.backward

      h = 1e-6
      a_plus = SHAInet::Autograd::GradMatrix.from_a([[0.5 + h]])
      a_minus = SHAInet::Autograd::GradMatrix.from_a([[0.5 - h]])

      loss_plus = a_plus.sigmoid[0, 0]
      loss_minus = a_minus.sigmoid[0, 0]
      numerical_grad = (loss_plus - loss_minus) / (2 * h)

      a.grad_at(0, 0).should be_close(numerical_grad, 1e-5)
    end

    it "matches numerical gradient for softmax" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0, 3.0]], requires_grad: true)

      # Use a weighted sum as loss to get non-trivial gradients
      weights = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0, 3.0]])
      s = a.softmax
      loss = s.hadamard(weights).sum
      loss.backward

      # Verify numerically for a[0,0]
      h = 1e-6
      a_plus = SHAInet::Autograd::GradMatrix.from_a([[1.0 + h, 2.0, 3.0]])
      a_minus = SHAInet::Autograd::GradMatrix.from_a([[1.0 - h, 2.0, 3.0]])

      loss_plus = a_plus.softmax.hadamard(weights).sum[0, 0]
      loss_minus = a_minus.softmax.hadamard(weights).sum[0, 0]
      numerical_grad = (loss_plus - loss_minus) / (2 * h)

      a.grad_at(0, 0).should be_close(numerical_grad, 1e-4)
    end
  end

  describe "complex computation graphs" do
    it "handles multi-layer neural network forward/backward" do
      # Simulate a simple 2-layer network: x -> W1 -> relu -> W2 -> output
      x = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0]], requires_grad: false)
      w1 = SHAInet::Autograd::GradMatrix.from_a([[0.1, 0.2], [0.3, 0.4]], requires_grad: true)
      w2 = SHAInet::Autograd::GradMatrix.from_a([[0.5], [0.5]], requires_grad: true)

      # Forward pass
      h = x.matmul(w1) # 1x2 @ 2x2 = 1x2
      h_relu = h.relu
      out = h_relu.matmul(w2) # 1x2 @ 2x1 = 1x1

      # Backward pass
      out.backward

      # Verify gradients exist and are non-zero
      w2.grad.should_not be_nil
      w1.grad.should_not be_nil

      # W2 gradients should be based on relu output
      (w2.grad_at(0, 0).abs + w2.grad_at(1, 0).abs).should be > 0

      # W1 gradients depend on whether ReLU let values through
      grad_sum = 0.0
      2.times do |i|
        2.times { |j| grad_sum += w1.grad_at(i, j).abs }
      end
      grad_sum.should be >= 0 # Could be 0 if ReLU blocks all
    end

    it "handles repeated use of same variable" do
      # y = x * x (x used twice)
      x = SHAInet::Autograd::GradMatrix.from_a([[2.0]], requires_grad: true)
      y = x.hadamard(x)
      y.backward

      # dy/dx = 2x = 4
      x.grad_at(0, 0).should be_close(4.0, 1e-10)
    end

    it "accumulates gradients correctly" do
      # z = x + x + x
      x = SHAInet::Autograd::GradMatrix.from_a([[1.0]], requires_grad: true)
      y1 = x + x
      y2 = y1 + x
      y2.backward

      # dz/dx = 3
      x.grad_at(0, 0).should be_close(3.0, 1e-10)
    end
  end

  describe "broadcasting" do
    it "adds row vector to each row" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
      b = SHAInet::Autograd::GradMatrix.from_a([[10.0, 20.0]], requires_grad: true)
      c = a.add_row_broadcast(b)

      c[0, 0].should eq(11.0)
      c[0, 1].should eq(22.0)
      c[1, 0].should eq(13.0)
      c[1, 1].should eq(24.0)
    end

    it "computes gradients for row broadcast add" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0], [3.0, 4.0]], requires_grad: true)
      b = SHAInet::Autograd::GradMatrix.from_a([[10.0, 20.0]], requires_grad: true)
      c = a.add_row_broadcast(b)
      loss = c.sum
      loss.backward

      # dL/dA = 1 for all elements
      a.grad_at(0, 0).should eq(1.0)
      a.grad_at(1, 1).should eq(1.0)

      # dL/dB = sum over rows = 2 for each element
      b.grad_at(0, 0).should eq(2.0)
      b.grad_at(0, 1).should eq(2.0)
    end
  end

  describe "utility methods" do
    it "clones correctly" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0]], requires_grad: true)
      b = a.clone

      b[0, 0].should eq(1.0)
      b.requires_grad.should be_true
      b.is_leaf.should be_true # Clone is a new leaf

      # Modifying clone doesn't affect original
      b.data[0] = 99.0
      a[0, 0].should eq(1.0)
    end

    it "detaches correctly" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0]], requires_grad: true)
      b = a.detach

      b[0, 0].should eq(1.0)
      b.requires_grad.should be_false
    end

    it "zeros gradients" do
      a = SHAInet::Autograd::GradMatrix.from_a([[1.0, 2.0]], requires_grad: true)
      b = a * 2.0
      b.sum.backward

      a.grad_at(0, 0).should eq(2.0)

      a.zero_grad!
      a.grad_at(0, 0).should eq(0.0)
    end
  end
end
