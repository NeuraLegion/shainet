require "./spec_helper"

describe SHAInet do
  # TODO: Write tests
  it "check functions" do
    # matrix1 = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
    # matrix2 = [[1, 1, 1, 1], [10, 10, 10, 10]]
    # new_m = SHAInet.dot_product(matrix1, matrix2)
    m1 = MatrixExtend::Matrix(Int32).new(3, 3, 1)
    m2 = MatrixExtend::Matrix(Int32).new(3, 2, 2)
    m3 = MatrixExtend::Matrix(Int32).new(3, 2, [[10, 10], [100, 100], [1000, 1000]])
    # m3 = [10, 100, 1000]
    pp m1.to_a, m2.to_a, m3.to_a
    m1xm2 = m1*m2
    pp m1xm2.to_a
    m1xm2xm3 = SHAInet.h_product(m1xm2, m3)
    pp m1xm2xm3.to_a

    # def each_line(&b)
    #   @data.each_with_index { |l, i| yield l, i }
    #   self
    # end
  end
end
