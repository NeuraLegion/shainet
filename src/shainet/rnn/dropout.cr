module SHAInet
  # Utility methods for dropout within RNN layers
  module RNNDropout
    # Returns a new Array where approximately `drop_percent` percent of values
    # from `array` have been set to 0.0. `drop_percent` should be between 0 and 100.
    def self.apply(array : Array(Float64), drop_percent : Int32)
      raise ArgumentError.new("drop_percent must be between 0 and 100") unless 0 <= drop_percent <= 100
      array.map do |v|
        rand(0...100) < drop_percent ? 0.0 : v
      end
    end
  end
end
