require "./spec_helper"

describe SHAInet do
  # TODO: Write tests
  it "check functions" do
    # expected = [1, 0.5]
    # actual = [0.5, 0.2]

    p SHAInet.linear_distance(expected = 1, actual = 0.5)
    puts "---"
    p SHAInet.squared_distance(expected = 1, actual = 0.5)
    puts "---"
    p SHAInet.cross_entropy(expected = 1, actual = 0.5)
  end
end
