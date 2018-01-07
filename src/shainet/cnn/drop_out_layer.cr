require "logger"

module SHAInet
  # class DO_layer
  #   property :neurons, :all_synapses
  #   getter output : Array(Array(Array(Float64))), :filters

  #   def initialize(@input_layer : CNN_layer, @drop_percent : Int32 = 5, @logger : Logger = Logger.new(STDOUT))
  #     @filters = input_layer.filters
  #     @output = input_layer.output
  #   end

  #   # Randomly select and deactivate a percentage of the neurons from the previous layer
  #   def activate(@input_layer : CNN_layer)
  #     @filters.size.times do |filter|
  #       @filters[filter].size.times do |row|
  #         @filters[filter][row].size.times do |col|
  #           x = rand(0..100)
  #           if x <= @drop_percent
  #             @filters[filter][row][col].activation = 0.0
  #             @output[filter][row][col] = 0.0
  #           end
  #         end
  #       end
  #     end
  #   end

  #   def inspect(what : String)
  #     case what
  #     when "weights"
  #       puts "input layer has no wights"
  #     when "bias"
  #     when "neurons"
  #     end
  #   end
  # end
end
