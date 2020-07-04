require "json"

module SHAInet
  class NetDump
    include JSON::Serializable
    property layers : Array(LayerDump)
  end

  class LayerDump
    include JSON::Serializable
    property l_type : String
    property neurons : Array(NeuronDump)
    property activation_function : String
  end

  class NeuronDump
    include JSON::Serializable
    property id : String
    property bias : Float64
    property n_type : String
    property synapses_in : Array(SynapseDump)
    property synapses_out : Array(SynapseDump)
  end

  class SynapseDump
    include JSON::Serializable
    property source : String
    property destination : String
    property weight : Float64
  end
end
