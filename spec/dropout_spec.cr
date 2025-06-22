require "./spec_helper"

describe SHAInet::DropoutLayer do
  it "drops approximately the given percentage of neurons" do
    input = SHAInet::InputLayer.new([10, 10, 1])
    dropout = SHAInet::DropoutLayer.new(input, 30)

    data = Array(Array(Array(Float64))).new(1) { Array(Array(Float64)).new(10) { Array(Float64).new(10, 1.0) } }
    input.activate(data)

    runs = 1000
    total_ratio = 0.0

    runs.times do
      dropout.activate
      dropped = 0
      total = 0
      dropout.filters.each do |filter|
        filter.neurons.each do |row|
          row.each do |neuron|
            total += 1
            dropped += 1 if neuron.activation == 0.0
          end
        end
      end
      total_ratio += dropped.to_f / total
    end

    average = total_ratio / runs
    (average).should be_close(0.30, 0.05)
  end
end
