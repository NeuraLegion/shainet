require "./spec_helper"

private def cpu_softmax_cross_entropy_label(logits : SHAInet::SimpleMatrix, labels : Array(Int32))
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
      grad[i, j] = Math.exp(logits[i, j] - max) / sum
    end
    label = labels[i]
    p = grad[i, label]
    grad[i, label] = p - 1.0
    loss += -Math.log(p.clamp(1e-15, 1.0))
  end
  {loss: loss, grad: grad}
end

describe "CUDA softmax cross entropy with labels" do
  it "matches CPU implementation" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    logits = SHAInet::SimpleMatrix.from_a([[1.0, 2.0, 0.5], [0.1, -1.0, 0.3]])
    labels = [1, 0]
    ref = cpu_softmax_cross_entropy_label(logits, labels)

    g_pred = SHAInet::GPUMemory.to_gpu(logits).as(SHAInet::CudaMatrix)
    g_labels = SHAInet::CudaMatrix.new(labels.size, 1)
    labels.each_with_index { |l, i| g_labels[i, 0] = l.to_f64 }
    g_labels.sync_to_device!
    grad = SHAInet::CudaMatrix.new(logits.rows, logits.cols)
    loss_val = 0.0
    {% if flag?(:enable_cuda) %}
      SHAInet::CUDNN.softmax_cross_entropy_label_loss_and_gradient(g_pred, g_labels, pointerof(loss_val), grad)
      grad.sync_from_device!
    {% else %}
      pending "CUDA not enabled"
    {% end %}

    loss_val.should be_close(ref[:loss], 1e-6)
    grad.rows.times do |i|
      grad.cols.times do |j|
        grad[i, j].should be_close(ref[:grad][i, j], 1e-6)
      end
    end
  end
end
