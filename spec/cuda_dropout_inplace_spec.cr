require "./spec_helper"

describe "CUDA dropout! in-place" do
  it "masks values on the GPU" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    mat = SHAInet::CudaMatrix.new(32, 32)
    mat.fill!(1.0)

    # Apply dropout in-place without CPU sync
    mat.dropout!(0.5)

    # GPU memory should now contain zeros in some positions
    mat.sync_from_device!
    zero_count = 0
    mat.rows.times do |i|
      mat.cols.times do |j|
        zero_count += 1 if mat[i, j] == 0.0
      end
    end
    zero_count.should be > 0
  end
end
