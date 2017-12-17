require "./spec_helper"

describe SHAInet::Layer do
  # TODO: Write tests
  # it "check layer creation" do
  #   # Layer creation needs: layer_size, layer_type, memory_size
  #   memory_layer = SHAInet::Layer.new(4, :memory, 2)
  #   eraser_layer = SHAInet::Layer.new(4, :eraser, 2)

  #   pp memory_layer
  #   pp eraser_layer
  #   # payloads_v.each { |x| puts x }
  # end

  it "check layer added functions" do
    # Layer creation needs: layer_type, layer_size, memory_size
    layer = SHAInet::Layer.new(:memory, 3)
    pp layer
    puts "------------------"

    layer.random_seed
    pp layer
    puts "------------------"

    layer.memory_change(2)
    pp layer
    puts "------------------"

    layer.type_change(:eraser)
    pp layer
  end
end
