require "./spec_helper"

describe "load_from_pt for transformer" do
  it "loads a minimal transformer" do
    model_path = "spec/tmp_transformer.pt"
    system("python3", ["scripts/build_transformer_model.py", model_path])

    net = SHAInet::Network.new
    net.load_from_pt(model_path)
    out = net.run([1])
    out.size.should eq(2)
    File.delete(model_path)
  end
end
