require "./spec_helper"
require "csv"

# Extract train data
system("cd #{__DIR__}/test_data && tar xvf tests.tar.gz")

describe SHAInet::CNN do
  it "Check basic cnn features" do
    data = [
      [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
    ]

    # input_volume, filters, filter_size, stride, padding
    cnn = SHAInet::Conv_layer.new(input_volume = [6, 6, 2], filters = 3, filter_size = 2, stride = 1, padding = 2)
    # p cnn.input
    # cnn.filters.each { |row| pp row }
    pp cnn.output
    # cnn.convolve(data)
  end
end
