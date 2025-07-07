require "./spec_helper"

private def cpu_softmax_cross_entropy(logits : SHAInet::SimpleMatrix, target : SHAInet::SimpleMatrix)
  rows = logits.rows
  cols = logits.cols
  grad = SHAInet::SimpleMatrix.zeros(rows, cols)
  loss = 0.0
  rows.times do |i|
    max = -Float64::INFINITY
    cols.times { |j| max = Math.max(max, logits[i, j]) }
    sum = 0.0
    cols.times { |j| sum += Math.exp(logits[i, j] - max) }
    cols.times do |j|
      p = Math.exp(logits[i, j] - max) / sum
      t = target[i, j]
      grad[i, j] = p - t
      loss += -t * Math.log(p.clamp(1e-15, 1.0))
    end
  end
  {loss: loss, grad: grad}
end

describe "CUDA softmax cross entropy" do
  it "matches CPU implementation" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    logits = SHAInet::SimpleMatrix.from_a([[1.0, 2.0, 0.5], [0.1, -1.0, 0.3]])
    target = SHAInet::SimpleMatrix.from_a([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    ref = cpu_softmax_cross_entropy(logits, target)

    g_pred = SHAInet::GPUMemory.to_gpu(logits).as(SHAInet::CudaMatrix)
    g_target = SHAInet::GPUMemory.to_gpu(target).as(SHAInet::CudaMatrix)
    grad = SHAInet::CudaMatrix.new(logits.rows, logits.cols)
    loss_val = 0.0
    SHAInet::CUDNN.softmax_cross_entropy_loss_and_gradient(g_pred, g_target, pointerof(loss_val), grad)
    grad.sync_from_device!

    loss_val.should be_close(ref[:loss], 1e-6)
    grad.rows.times do |i|
      grad.cols.times do |j|
        grad[i, j].should be_close(ref[:grad][i, j], 1e-6)
      end
    end
  end

  it "computes correctly on CPU when CUDA is disabled" do
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    logits = SHAInet::SimpleMatrix.from_a([[1.0, 0.0, -1.0]])
    target = SHAInet::SimpleMatrix.from_a([[0.0, 0.0, 1.0]])
    result = cpu_softmax_cross_entropy(logits, target)
    # For target class 3 the expected cross entropy is around 2.4076
    result[:loss].should be_close(2.407605, 1e-5)
    ENV.delete("SHAINET_DISABLE_CUDA")
  end
end
