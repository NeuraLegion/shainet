require "./spec_helper"

describe "Matrix Pipeline Optimization" do
  it "maintains matrix type consistency throughout transformer pipeline" do
    # Test matrix type consistency
    mat_klass = SHAInet::CUDA.fully_available? ? SHAInet::CudaMatrix : SHAInet::SimpleMatrix

    # Test MultiHeadAttention
    d_model = 8
    num_heads = 2
    mha = SHAInet::MultiHeadAttention.new(d_model, num_heads)

    input_seq = mat_klass.new(2, d_model).random_fill!
    output = mha.forward(input_seq)

    # Test that operations complete successfully
    output.should_not be_nil
    output.rows.should eq(input_seq.rows)
    output.cols.should eq(input_seq.cols)

    # Test PositionWiseFF
    ff = SHAInet::PositionWiseFF.new(d_model, 16)
    ff_output = ff.forward(output)

    # Test that operations complete successfully
    ff_output.should_not be_nil
    ff_output.rows.should eq(output.rows)
    ff_output.cols.should eq(output.cols)

    # Test LayerNorm
    ln = SHAInet::LayerNorm.new(d_model)
    ln_output = ln.forward(ff_output)

    # Test that operations complete successfully
    ln_output.should_not be_nil
    ln_output.rows.should eq(ff_output.rows)
    ln_output.cols.should eq(ff_output.cols)

    # Test that all matrices are valid matrix types
    [input_seq, output, ff_output, ln_output].each do |matrix|
      (matrix.is_a?(SHAInet::SimpleMatrix) || matrix.is_a?(SHAInet::CudaMatrix)).should be_true
    end
  end

  it "uses GPU kernels when available with proper fallbacks" do
    d_model = 16

    # Test that operations work regardless of whether CUDA kernels are available
    if SHAInet::CUDA.fully_available?
      input = SHAInet::CudaMatrix.new(4, d_model).random_fill!
      input.should be_a(SHAInet::CudaMatrix)
    else
      input = SHAInet::SimpleMatrix.new(4, d_model).random_fill!
      input.should be_a(SHAInet::SimpleMatrix)
    end

    # Test LayerNorm with fallback handling
    ln = SHAInet::LayerNorm.new(d_model)

    # This should not raise an error regardless of kernel availability
    result = ln.forward(input)
    result.should_not be_nil
    result.rows.should eq(input.rows)
    result.cols.should eq(input.cols)

    # Result should be a valid matrix type
    (result.is_a?(SHAInet::SimpleMatrix) || result.is_a?(SHAInet::CudaMatrix)).should be_true
  end
end
