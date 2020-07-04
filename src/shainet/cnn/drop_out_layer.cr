require "log"

module SHAInet
  class DropoutLayer
    getter filters : Array(Filter), drop_percent : Int32, prev_layer : CNNLayer | ConvLayer

    # Drop percent is an Integer, i.e 5 is 5%
    def initialize(@prev_layer : CNNLayer | ConvLayer,
                   @drop_percent : Int32 = 5,
                   @log : Log = Log.new(STDOUT))
      #
      raise CNNInitializationError.new("Drop percent must be Int of 0-100") unless (0..100).includes?(@drop_percent)
      @filters = @prev_layer.filters.clone # Volume of this layer is the same as the previus layer
    end

    # Randomly select and deactivate a percentage of the neurons from the previous layer
    def activate
      @filters.size.times do |filter|
        @filters[filter].neurons.size.times do |row|
          @filters[filter].neurons[row].size.times do |neuron|
            x = rand(0..100)
            if x <= @drop_percent
              @filters[filter].neurons[row][neuron].activation = 0.0
            else
              @filters[filter].neurons[row][neuron].activation = @prev_layer.filters[filter].neurons[row][neuron].activation
            end
          end
        end
      end
    end

    # Send the gradients from current layer backwards without weights
    def error_prop(batch : Bool = false)
      @filters.size.times do |filter|
        @filters[filter].neurons.size.times do |row|
          @filters[filter].neurons[row].size.times do |neuron|
            @prev_layer.filters[filter].neurons[row][neuron].gradient = @filters[filter].neurons[row][neuron].gradient
          end
        end
      end
    end

    def update_wb(learn_type : Symbol | String, batch : Bool = false)
      # Do nothing
    end

    def inspect(what : String)
      puts "##################################################"
      puts "DropoutLayer:"
      puts "----------"
      case what
      when "weights"
        puts "Drop-out layer has no weights"
      when "bias"
        puts "Drop-out layer has no bias"
      when "activations"
        @filters.each_with_index do |filter, f|
          puts "---"
          puts "Filter: #{f}, neuron activations are:"
          filter.neurons.each do |row|
            puts "#{row.map { |n| n.activation.round(4) }}"
          end
        end
      when "gradients"
        @filters.each_with_index do |filter, f|
          puts "---"
          puts "Filter: #{f}, neuron gradients are:"
          filter.neurons.each do |row|
            puts "#{row.map { |n| n.gradient.round(4) }}"
          end
        end
      end
      puts "------------------------------------------------"
    end
  end
end
