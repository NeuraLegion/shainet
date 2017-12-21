require "./spec_helper"
require "matrix_extend"

describe SHAInet do
  # TODO: Write tests
  it "check functions" do
    # matrix1 = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
    # matrix2 = [[1, 1, 1, 1], [10, 10, 10, 10]]
    # new_m = SHAInet.dot_product(matrix1, matrix2)
    m1 = MatrixExtend::Matrix(Int32).new(3, 2, 1)
    m2 = MatrixExtend::Matrix(Int32).new(2, 3, 2)
    new_m = m1*m2
    p m1
    p m2
    p new_m
    # p new_m

  end
end
