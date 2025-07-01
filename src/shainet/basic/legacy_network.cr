module SHAInet
  class LegacyNetwork
    getter network : Network

    def initialize
      @network = Network.new
    end

    forward_missing_to @network
  end
end
