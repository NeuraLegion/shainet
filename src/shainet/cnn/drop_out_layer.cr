require "logger"

module SHAInet
  class DropoutLayer
    getter filters : Array(Array(Array(Array(Neuron)))) | Array(Filter), drop_percent : Int32, prev_layer : CNNLayer | ConvLayer
    property next_layer : CNNLayer | ConvLayer | DummyLayer

    # Calls different activaton based on previous layer type
    def activate
      _activate(@prev_layer)
    end

    #################################################
    # # This part is for dealing with conv layers # #

    # Drop percent is an Int, i.e 5 is 5%
    def initialize(prev_layer : ConvLayer, @drop_percent : Int32 = 5, @logger : Logger = Logger.new(STDOUT))
      raise CNNInitializationError.new("Drop percent must be Int of 0-100") unless (0..100).includes?(@drop_percent)

      @filters = prev_layer.filters.clone
      @prev_layer = prev_layer
      @next_layer = DummyLayer.new
      @prev_layer.next_layer = self
    end

    # Randomly select and deactivate a percentage of the neurons from the previous layer
    def _activate(prev_layer : ConvLayer)
      @filters = prev_layer.filters.clone
      @filters.each do |filter|
        filter.neurons.each do |row|
          row.each do |neuron|
            x = rand(0..100)
            if x <= @drop_percent
              neuron.activation = 0.0
            end
          end
        end
      end
    end

    #######################################################################
    # # This part is for dealing with all layers other than conv layers # #

    # Drop percent is an Int, i.e 5 is 5%
    def initialize(prev_layer : CNNLayer, @drop_percent : Int32, @logger : Logger = Logger.new(STDOUT))
      raise CNNInitializationError.new("Drop percent must be Int of 0-100") unless (0..100).includes?(@drop_percent)

      @filters = prev_layer.filters.clone
      @prev_layer = prev_layer
      @next_layer = DummyLayer.new
      @prev_layer.next_layer = self
    end

    def _activate(prev_layer : CNNLayer)
      @filters = prev_layer.filters.clone
      @filters.first.each do |channel|
        channel.each do |row|
          row.each do |neuron|
            x = rand(0..100)
            if x <= @drop_percent
              neuron.activation = 0.0
            end
          end
        end
      end
    end

    def error_prop
      _error_prop(@next_layer)
    end

    def _error_prop(next_layer : FullyConnectedLayer)
      @filters.each do |filter|
        filter.each do |channel|
          channel.each do |row|
            row.each { |neuron| neuron.hidden_error_prop }
          end
        end
      end
    end

    def _error_prop(next_layer : MaxPoolLayer)
      @filters.each_with_index do |_f, filter|
        _f.each_with_index do |_ch, channel|
          input_x = input_y = output_x = output_y = 0

          while input_y < (@filters[filter][channel].size - @pool + @stride)   # Break out of y
            while input_x < (@filters[filter][channel].size - @pool + @stride) # Break out of x (assumes x = y)
              pool_neuron = next_layer.filters[filter][channel][output_y][output_x]

              # Only propagate error to the neurons that were chosen during the max pool
              @filters[filter][channel][input_y..(input_y + @pool - 1)].each do |row|
                row[input_x..(input_x + @pool - 1)].each do |neuron|
                  if neuron.activation == pool_neuron.activation
                    neuron.gradient = pool_neuron.gradient
                  end
                end
              end

              input_x += @stride
              output_x += 1
            end
            input_x = output_x = 0
            input_y += @stride
            output_y += 1
          end
        end
      end
    end

    def _error_prop(next_layer : ReluLayer | DropoutLayer)
      @filters.each_with_index do |filter, fi|
        filter.each_with_index do |channel, ch|
          channel.each_with_index do |row, r|
            row.each_with_index do |neuron, n|
              neuron.gradient = next_layer.filters[fi][ch][r][n].gradient
            end
          end
        end
      end
    end

    def _error_prop(next_layer : DummyLayer)
      # Do nothing because this is the last layer in the network
    end

    def inspect(what : String)
      puts "Drop-out layer:"
      case what
      when "weights"
        puts "Drop-out layer has no wights"
      when "bias"
        puts "Drop-out layer has no bias"
      when "activations"
        @filters.each_with_index do |filter, f|
          puts "Filter: #{f}"
          filter.each_with_index do |channel, ch|
            puts "Channel: #{ch}, neuron activations are:"
            channel.each do |row|
              puts "#{row.map { |n| n.activation }}"
            end
          end
        end
      end
      puts "------------"
    end
  end
end
