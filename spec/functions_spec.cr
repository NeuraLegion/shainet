require "./spec_helper"

describe SHAInet do
  # TODO: Write tests
  it "check sigmoid" do
    ((0..1).includes?(SHAInet.sigmoid(0.5))).should eq(true)
  end

  it "check bp_sigmoid" do
    ((-1..1).includes?(SHAInet.bp_sigmoid(0.5))).should eq(true)
  end

  it "check log_sigmoid" do
    ((0..1).includes?(SHAInet.log_sigmoid(0.5))).should eq(true)
  end

  it "check tanh" do
    ((-1..1).includes?(SHAInet.tanh(0.5))).should eq(true)
  end

  it "check relu" do
    ((0..Int64::MAX).includes?(SHAInet.relu(0.5))).should eq(true)
  end

  it "check l_relu" do
    ((Int64::MIN..Int64::MAX).includes?(SHAInet.l_relu(0.5, 0.5))).should eq(true)
  end
end
