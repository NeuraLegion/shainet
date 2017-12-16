require "./spec_helper"

describe SHAInet::Layer do
  # TODO: Write tests
  it "check layer creation" do
    # Layer creation needs: layer_size, layer_type, memory_size
    memory_layer = SHAInet::Layer.new(4, :memory, 2)
    eraser_layer = SHAInet::Layer.new(4, :eraser, 2)

    pp memory_layer
    pp eraser_layer
    # payloads_v.each { |x| puts x }
  end

  it "check layer added functions" do
    layer = SHAInet::Layer.new(3, :memory, 2)
    # pp layer

    layer.random_seed
    pp layer

    layer.memory_change(1)
    pp layer

    layer.type_change(:eraser)
    pp layer
  end
end
