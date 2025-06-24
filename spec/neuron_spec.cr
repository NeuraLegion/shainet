require "./spec_helper"

describe SHAInet::Neuron do
  puts "############################################################"
  it "Initialize neuron" do
    puts "\n"
    SHAInet::NEURON_TYPES.each do |type|
      neuron = SHAInet::Neuron.new(type)
      neuron.should be_a(SHAInet::Neuron)
    end
  end

  it "propagates based on neuron type" do
    puts "\n"
    src = SHAInet::Neuron.new("memory")
    dest = SHAInet::Neuron.new("memory")

    syn = SHAInet::Synapse.new(src, dest)
    syn.weight = 1.0
    src.activation = 1.0
    syn.propagate_forward.should eq(1.0)

    src.n_type = "eraser"
    syn.propagate_forward.should eq(-1.0)

    src.n_type = "amplifier"
    syn.propagate_forward.should eq(2.0)

    src.n_type = "fader"
    syn.propagate_forward.should eq(0.5)

    src.n_type = "sensor"
    syn.propagate_forward.should eq(1.0)
  end
end
