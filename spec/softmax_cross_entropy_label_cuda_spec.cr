require "./spec_helper"

# Use SHAInet.softmax for reference, matching main code logic
private def cpu_softmax_cross_entropy_label(logits : SHAInet::SimpleMatrix, labels : Array(Int32))
  rows = logits.rows
  cols = logits.cols
  grad = SHAInet::SimpleMatrix.zeros(rows, cols)
  loss = 0.0
  logits_rows = logits.to_a
  rows.times do |i|
    probs = SHAInet.softmax(logits_rows[i])
    label = labels[i]
    row_loss = -Math.log(probs[label].clamp(1e-15, 1.0))
    puts "[CPU] Row \\#{i}: logits=\\#{logits_rows[i].inspect}, probs=\\#{probs.inspect}, label=\\#{label}, row_loss=\\#{row_loss}"
    loss += row_loss
    cols.times do |j|
      grad[i, j] = probs[j]
    end
    grad[i, label] -= 1.0
  end
  puts "[CPU] Total loss: \\#{loss}"
  {loss: loss, grad: grad}
end

describe "CUDA softmax cross entropy with labels" do
  it "matches CPU implementation" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    logits = SHAInet::SimpleMatrix.from_a([[1.0, 2.0, 0.5], [0.1, -1.0, 0.3]])
    labels = [1, 0]
    ref = cpu_softmax_cross_entropy_label(logits, labels)

    # Ensure logits are synced to device and not overwritten
    g_pred = SHAInet::GPUMemory.to_gpu(logits).as(SHAInet::CudaMatrix)
    g_pred.sync_to_device! unless g_pred.device_dirty? == false

    # Prepare labels as Int32 array on device
    label_ids = labels.map(&.to_i)
    bytes = (label_ids.size * 4).to_u64
    labels_dev = Pointer(Int32).null
    SHAInet::CUDA.malloc(pointerof(labels_dev).as(Pointer(Pointer(Void))), bytes)
    SHAInet::CUDA.memcpy(labels_dev.as(Pointer(Void)), label_ids.to_unsafe.as(Pointer(Void)), bytes, SHAInet::CUDA::MemcpyKind::HostToDevice)

    grad = SHAInet::CudaMatrix.new(logits.rows, logits.cols)
    loss_val = 0.0
    {% if flag?(:enable_cuda) %}
      # Call the CUDA kernel directly, not via CUDNN wrapper
      SHAInet::CUDA.softmax_cross_entropy_label(
        g_pred.device_ptr.not_nil!,
        labels_dev,
        grad.device_ptr.not_nil!,
        pointerof(loss_val),
        g_pred.rows,
        g_pred.cols
      )
      grad.mark_device_dirty!
      grad.sync_from_device!
      SHAInet::CUDA.free(labels_dev.as(Pointer(Void))) unless labels_dev.null?
    {% else %}
      pending "CUDA not enabled"
    {% end %}

    puts "[CUDA] loss_val: \\#{loss_val}"
    grad.sync_from_device!
    grad_arr = grad.to_a
    grad_arr.each_with_index do |row, i|
      puts "[CUDA] Row \\#{i}: grad=\\#{row.inspect}"
    end
    loss_val.should be_close(ref[:loss], 1e-6)
    grad.rows.times do |i|
      grad.cols.times do |j|
        grad[i, j].should be_close(ref[:grad][i, j], 1e-6)
      end
    end
  end
end
