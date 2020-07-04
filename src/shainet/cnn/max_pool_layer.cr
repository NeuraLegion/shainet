require "log"

module SHAInet
  class MaxPoolLayer
    Log = ::Log.for(self)
    getter filters : Array(Filter), pool : Int32, stride : Int32, prev_layer : CNNLayer | ConvLayer

    def initialize(@prev_layer : CNNLayer | ConvLayer,
                   @pool : Int32,
                   @stride : Int32)
      #
      prev_w = prev_layer.filters.first.neurons.size # Assumes row == height
      new_w = ((prev_w.to_f64 - @pool.to_f64)/@stride.to_f64 + 1).to_f64
      raise CNNInitializationError.new("Max pool layer parameters are incorrect, change pool size or stride") unless new_w.to_i == new_w

      filters = @prev_layer.filters.size
      width = height = new_w.to_i # Assumes row == height

      @filters = Array(Filter).new(filters) { Filter.new([width, height, 1]) }
    end

    def activate
      @prev_layer.filters.size.times do |filter|
        # Zoom in on a small window out of the data matrix and update
        input_x = input_y = output_x = output_y = 0

        while input_y < (@prev_layer.filters[filter].neurons.size - @pool + @stride)         # Break out of y
          while input_x < (@prev_layer.filters[filter].neurons.first.size - @pool + @stride) # Break out of x
            window = @prev_layer.filters[filter].neurons[input_y..(input_y + @pool - 1)].map { |row| row[input_x..(input_x + @pool - 1)].map { |neuron| neuron.activation } }
            @filters[filter].neurons[output_y][output_x].activation = window.flatten.max

            input_x += @stride
            output_x += 1
          end
          input_x = output_x = 0
          input_y += @stride
          output_y += 1
        end
      end
    end

    # Send the gradients from current layer backwards without weights, only to the max neurons from forward prop
    def error_prop(batch : Bool = false)
      @prev_layer.filters.size.times do |filter|
        # Zoom in on a small window out of the data matrix and update
        input_x = input_y = output_x = output_y = 0

        while input_y < (@prev_layer.filters[filter].neurons.size - @pool + @stride)         # Break out of y
          while input_x < (@prev_layer.filters[filter].neurons.first.size - @pool + @stride) # Break out of x
            window = @prev_layer.filters[filter].neurons[input_y..(input_y + @pool - 1)].map { |row| row[input_x..(input_x + @pool - 1)] }
            pool_neuron = @filters[filter].neurons[output_y][output_x] # Neuron from this layer

            # Only propagate error to the neurons that were chosen during the max pool
            window.each do |row|
              row.each do |neuron|
                if neuron.activation == pool_neuron.activation
                  neuron.gradient = pool_neuron.gradient
                else
                  neuron.gradient = 0.0
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

    def update_wb(learn_type : Symbol | String, batch : Bool = false)
      # Do nothing
    end

    def inspect(what : String)
      puts "##################################################"
      puts "MaxPoolLayer:"
      puts "----------"
      case what
      when "weights"
        puts "Maxpool layer has no weights"
      when "bias"
        puts "Maxpool layer has no bias"
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
