require "./spec_helper"

describe SHAInet::Network do
  it "loads a simple TorchScript model" do
    tmp = File.tempfile("model", ".pt")
    model = tmp.path
    tmp.close
    build_err = IO::Memory.new
    status = Process.run(
      "python3",
      ["-W", "ignore", "#{__DIR__}/../scripts/build_simple_model.py", model],
      error: build_err
    )
    status.success?.should eq(true), build_err.to_s
    net = SHAInet::Network.new
    expect_out = IO::Memory.new
    expect_err = IO::Memory.new
    status = Process.run(
      "python3",
      ["-W", "ignore", "#{__DIR__}/../scripts/pt_forward.py", model],
      output: expect_out,
      error: expect_err
    )
    status.success?.should eq(true), expect_err.to_s
    expected = expect_out.to_s.to_f

    net.load_from_pt(model)
    output = net.run([1.0, 2.0]).first
    (output - expected).abs.should be < 1e-3
    File.delete(model)
  end
end
