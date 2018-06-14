require "json"

module SHAInet
  class NetDump
    JSON.mapping({
      layers: Array(LayerDump),
    })
  end

  class LayerDump
    JSON.mapping({
      l_type:              String,
      neurons:             Array(NeuronDump),
      activation_function: String,
    })
  end

  class NeuronDump
    JSON.mapping({
      id:           String,
      bias:         Float64,
      n_type:       String,
      synapses_in:  Array(SynapseDump),
      synapses_out: Array(SynapseDump),
    })
  end

  class SynapseDump
    JSON.mapping({
      source:      String,
      destination: String,
      weight:      Float64,
    })
  end
end
