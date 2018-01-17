require "logger"

module SHAInet
  class ReluLayer
    getter filters : Array(Filter), l_relu_slope : Float64, prev_layer : CNNLayer | ConvLayer

    # Add slope to initialize as leaky relu
    def initialize(@prev_layer : CNNLayer | ConvLayer,
                   @l_relu_slope : Float64 = 0.0,
                   @logger : Logger = Logger.new(STDOUT))
      #
      @filters = @prev_layer.filters.clone # Volume of this layer is the same as the previus layer
    end

    # Go over all neurons of previous layer and apply ReLu or leaky ReLu non-linearity
    def activate
      @filters.size.times do |filter|
        @filters[filter].neurons.size.times do |row|
          @filters[filter].neurons[row].size.times do |neuron|
            if @l_relu_slope == 0.0
              @filters[filter].neurons[row][neuron].activation = SHAInet._relu(@prev_layer.filters[filter].neurons[row][neuron].activation)
            else
              @filters[filter].neurons[row][neuron].activation = SHAInet._l_relu(@prev_layer.filters[filter].neurons[row][neuron].activation)
            end
          end
        end
      end
    end

    # Send the gradients from current layer backwards without weights
    def error_prop
      @filters.size.times do |filter|
        @filters[filter].neurons.size.times do |row|
          @filters[filter].neurons[row].size.times do |neuron|
            @prev_layer.filters[filter].neurons[row][neuron].gradient = @filters[filter].neurons[row][neuron].gradient
          end
        end
      end
    end

    def inspect(what : String)
      case what
      when "weights"
        puts "ReLu layer has no weights"
      when "bias"
        puts "ReLu layer has no bias"
      when "activations"
        @filters.each_with_index do |filter, f|
          puts "---"
          puts "Filter: #{f}, neuron activations are:"
          filter.neurons.each do |row|
            puts "#{row.map { |n| n.activation.round(4) }}"
          end
        end
      end
      puts "------------------------------------------------"
    end
  end
end
