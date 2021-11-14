require "json"

module SHAInet
  class NetDump
    include JSON::Serializable

    layers : Array(LayerDump)
  end

  class LayerDump
    include JSON::Serializable

    l_type : String
    neurons : Array(NeuronDump)
    activation_function : String
  end

  class NeuronDump
    include JSON::Serializable

    id : String
    bias : Float64
    n_type : String
    synapses_in : Array(SynapseDump)
    synapses_out : Array(SynapseDump)
  end

  class SynapseDump
    include JSON::Serializable

    source : String
    destination : String
    weight : Float64
  end
end
