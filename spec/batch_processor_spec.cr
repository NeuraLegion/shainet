require "./spec_helper"

describe SHAInet::BatchProcessor do
  it "matches individual layer forward passes" do
    mat_klass = SHAInet::CUDA.available? ? SHAInet::CudaMatrix : SHAInet::SimpleMatrix
    layer1 = SHAInet::MatrixLayer.new(2, 3)
    layer2 = SHAInet::MatrixLayer.new(2, 3)

    input1 = mat_klass.from_a([[1.0, 2.0]])
    input2 = mat_klass.from_a([[3.0, 4.0]])

    out1 = layer1.forward(input1)
    out2 = layer2.forward(input2)

    batch_out = SHAInet::BatchProcessor.process_batch([layer1, layer2], [input1, input2])

    batch_out.size.should eq 2

    [out1, out2].each_with_index do |expected, idx|
      result = batch_out[idx]
      expected.rows.times do |i|
        expected.cols.times do |j|
          result[i, j].should be_close(expected[i, j], 1e-6)
        end
      end
    end
  end
end
