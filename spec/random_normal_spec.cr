require "./spec_helper"
require "csv"

describe SHAInet::RandomNormal do
  puts "############################################################"

  it "Get 10000 random samples from normal distribution" do
    puts "\n"

    data = SHAInet::RandomNormal.sample(n: 10000, mu: 5.0, sigma: 2.0)
    csv_file = CSV.build do |csv|
      data.each do |value|
        csv.row value
      end
    end

    # Save to file to visualize distribution
    # File.write("csv_file.csv", csv_file)
  end
end
