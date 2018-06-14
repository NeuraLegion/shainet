require "./spec_helper"

describe SHAInet do
  puts "############################################################"
  it "check sigmoid" do
    puts "\n"
    ((0..1).includes?(SHAInet.sigmoid.call(0.5).first)).should eq(true)
  end

  puts "############################################################"
  it "check bp_sigmoid" do
    puts "\n"
    ((-1..1).includes?(SHAInet.bp_sigmoid.call(0.5).first)).should eq(true)
  end

  puts "############################################################"
  it "check log_sigmoid" do
    puts "\n"
    ((0..1).includes?(SHAInet.log_sigmoid.call(0.5).first)).should eq(true)
  end

  puts "############################################################"
  it "check tanh" do
    puts "\n"
    ((-1..1).includes?(SHAInet.tanh.call(0.5).first)).should eq(true)
  end

  puts "############################################################"
  it "check relu" do
    puts "\n"
    ((0..Int64::MAX).includes?(SHAInet.relu.call(0.5).first)).should eq(true)
  end

  puts "############################################################"
  it "check l_relu" do
    puts "\n"
    ((Int64::MIN..Int64::MAX).includes?(SHAInet.l_relu.call(0.5).first)).should eq(true)
  end

  puts "############################################################"
  # it "check cross entropy" do
  puts "\n"

  #   puts SHAInet.cross_entropy_cost(0.0, 1.0)
  # end

  puts "############################################################"
  it "check softmax" do
    puts "\n"
    array = [1, 2, 3, 4]
    sf_array = SHAInet.softmax(array)
    puts sf_array
    puts "Array sum: #{sf_array.sum}"
  end

  puts "############################################################"
  it "check log_softmax" do
    puts "\n"
    array = [1, 2, 3, 4]
    sf_array = SHAInet.log_softmax(array)
    puts sf_array
    puts "Array sum: #{sf_array.sum}"
  end
end
