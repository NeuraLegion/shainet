require "./spec_helper"

describe SHAInet::Network do
  # TODO: Write tests

  it "figure out xor" do
    # This is testing to see if it works
    xor = SHAInet::Network.new(2, 10, 1)

    10000.times do
      xor.train([0, 0], [0])
      xor.train([1, 0], [1])
      xor.train([0, 1], [1])
      xor.train([1, 1], [0])
    end

    xor.feed_forward([0, 0])
    (xor.current_outputs.first < 0.1 && xor.current_outputs.first > -0.1).should eq(true)
  end
end
