require "logger"

module SHAInet
  class Filter
    getter input_surface : Array(Int32), window_size : Int32, stride : Int32, padding : Int32, activation_function : Proc(GenNum, Array(Float64))
    property neurons : Array(Array(Neuron)), synapses : Array(Array(Array(CnnSynapse))), bias : Float64

    def initialize(@input_surface : Array(Int32), # expecting [width, height, channels]
                   @padding : Int32 = 0,
                   @window_size : Int32 = 1,
                   @stride : Int32 = 1,
                   @activation_function : Proc(GenNum, Array(Float64)) = SHAInet.none)
      #
      @neurons = Array(Array(Neuron)).new(input_surface[1]) {
        Array(Neuron).new(input_surface[0]) { Neuron.new("memory") }
      }

      @synapses = Array(Array(Array(CnnSynapse))).new(input_surface[2]) {
        Array(Array(CnnSynapse)).new(@window_size) {
          Array(CnnSynapse).new(@window_size) { CnnSynapse.new }
        }
      }

      @bias = rand(-1..1).to_f64
      @blank_neuron = Neuron.new("memory") # This is needed for padding
      @blank_neuron.activation = 0.0
      @blank_neuron.gradient = 1.0
    end

    def propagate_forward(input_layer : ConvLayer | CNNLayer)
      # Starting locations
      input_x = input_y = -@padding
      output_x = output_y = 0

      # Takes a small window from the input data (Channel/Filter x Width x Height) to preform feed forward
      # Slides the window over the input data volume and updates each neuron of the filter
      # The window depth is the number of all channels/filters (depending on previous layer)
      while input_y < (input_layer.filters.first.neurons.size + @padding - @window_size + @stride)         # Break out of y
        while input_x < (input_layer.filters.first.neurons.first.size + @padding - @window_size + @stride) # Break out of x

          # Create the window from previous layer(x,y are shared across all channels)
          window = Array(Array(Array(Neuron))).new

          input_layer.filters.size.times do |channel|
            rows = Array(Array(Neuron)).new                      # Output xy matrix
            input_channel = input_layer.filters[channel].neurons # Input data xy matrix

            @window_size.times do |_row| # Iteration over y in the window
              input_row = input_y + _row

              # When dealing with top padding
              if input_row < 0
                rows << Array(Neuron).new(@window_size) { @blank_neuron }
                # puts "top pad"

                # When dealing with bottom padding
              elsif input_row > (input_channel.size - 1)
                rows << Array(Neuron).new(@window_size) { @blank_neuron }
                # puts "bottom pad"
              else
                row = Array(Neuron).new
                @window_size.times do |_col|
                  input_col = input_x + _col

                  # When dealing with left padding
                  if input_col < 0
                    row << @blank_neuron

                    # When dealing with right padding
                  elsif input_col > (input_channel.size - 1)
                    row << @blank_neuron

                    # When dealing with all other locations within the input data
                  else
                    row << input_channel[input_y + _row][input_x + _col]
                  end
                end
                rows << row
              end
            end
            window << rows
          end

          # # Gather the weighted activations from the entire window
          input_sum = Float64.new(0)
          @synapses.size.times do |channel|
            @synapses[channel].size.times do |row|
              @synapses[channel][row].size.times do |col| # Synapses are CnnSynpase in this case
              # Save the weighted activations from previous layer
                input_sum += @synapses[channel][row][col].weight*window[channel][row][col].activation
              end
            end
          end

          # # Add bias and apply activation function
          target_neuron = @neurons[output_y][output_x]
          target_neuron.input_sum = input_sum + @bias
          target_neuron.activation, target_neuron.sigma_prime = @activation_function.call(target_neuron.input_sum)

          # Go to next window horizontaly
          input_x += @stride
          output_x += 1
        end
        # Go to next window verticaly
        input_x = -@padding
        output_x = 0
        input_y += @stride
        output_y += 1
      end
    end

    def propagate_backward(input_layer : ConvLayer | CNNLayer)
      # Starting locations
      input_x = input_y = -@padding
      output_x = output_y = 0

      # Takes a small window from the input data (Channel/Filter x Width x Height) to preform feed forward
      # Slides the window over the input data volume and updates each neuron of the filter
      # The window depth is the number of all channels/filters (depending on previous layer)
      while input_y < (input_layer.filters.first.neurons.size + @padding - @window_size + @stride)         # Break out of y
        while input_x < (input_layer.filters.first.neurons.first.size + @padding - @window_size + @stride) # Break out of x

          # Create the window from previous layer(x,y are shared across all channels)
          window = Array(Array(Array(Neuron))).new

          input_layer.filters.size.times do |channel|
            rows = Array(Array(Neuron)).new                      # Output xy matrix
            input_channel = input_layer.filters[channel].neurons # Input data xy matrix

            @window_size.times do |_row| # Iteration over y in the window
              input_row = input_y + _row

              # When dealing with top padding
              if input_row < 0
                rows << Array(Neuron).new(@window_size) { @blank_neuron }
                # puts "top pad"

                # When dealing with bottom padding
              elsif input_row > (input_channel.size - 1)
                rows << Array(Neuron).new(@window_size) { @blank_neuron }
                # puts "bottom pad"
              else
                row = Array(Neuron).new
                @window_size.times do |_col|
                  input_col = input_x + _col

                  # When dealing with left padding
                  if input_col < 0
                    row << @blank_neuron

                    # When dealing with right padding
                  elsif input_col > (input_channel.size - 1)
                    row << @blank_neuron

                    # When dealing with all other locations within the input data
                  else
                    row << input_channel[input_y + _row][input_x + _col]
                  end
                end
                rows << row
              end
            end
            window << rows
          end

          # Propagate the weighted errors backwards to the entire window
          source_neuron = @neurons[output_y][output_x]

          # input_sum = Float64.new(0)
          @synapses.size.times do |channel|
            @synapses[channel].size.times do |row|
              @synapses[channel][row].size.times do |col| # Synapses are CnnSynpase in this case
              # Save the weighted activations from previous layer
                synapse = @synapses[channel][row][col]
                target_neuron = window[channel][row][col]
                target_neuron.gradient = synapse.weight*source_neuron.gradient*target_neuron.sigma_prime
                synapse.gradient_sum += target_neuron.activation*source_neuron.gradient
              end
            end
          end

          # Go to next window horizontaly
          input_x += @stride
          output_x += 1
        end
        # Go to next window verticaly
        input_x = -@padding
        output_x = 0
        input_y += @stride
        output_y += 1
      end
    end

    def clone
      filter_old = self
      filter_new = Filter.new(filter_old.input_surface, filter_old.padding, filter_old.window_size, filter_old.stride, filter_old.activation_function)

      filter_new.neurons = filter_old.neurons.clone
      filter_new.synapses = filter_old.synapses.clone
      filter_new.bias = filter_old.bias
      return filter_new
    end
  end
end

class CnnSynapse
  property weight : Float64, gradient : Float64, gradient_sum : Float64, prev_weight : Float64
  property prev_gradient : Float64, prev_delta : Float64, prev_delta_w : Float64
  property m_current : Float64, v_current : Float64, m_prev : Float64, v_prev : Float64

  def initialize
    @weight = rand(-1.0..1.0).to_f64   # Weight of the synapse
    @gradient_sum = Float64.new(0)     # For backpropogation of ConvLayers
    @gradient = rand(-0.1..0.1).to_f64 # Error of the synapse with respect to cost function (dC/dW)
    @prev_weight = Float64.new(0)      # Needed for delta rule improvement (with momentum)

    # Parameters needed for Rprop
    @prev_gradient = 0.0
    @prev_delta = 0.1
    @prev_delta_w = 0.1

    # Parameters needed for Adam
    @m_current = Float64.new(0) # Current moment value
    @v_current = Float64.new(0) # Current moment**2 value
    @m_prev = Float64.new(0)    # Previous moment value
    @v_prev = Float64.new(0)    # Previous moment**2 value
  end

  def randomize_weight
    @weight = rand(-0.1..0.1).to_f64
  end

  def clone
    synapse_old = self
    synapse_new = CnnSynapse.new

    synapse_new.weight = synapse_old.weight
    synapse_new.gradient = synapse_old.gradient
    synapse_new.prev_weight = synapse_old.prev_weight
    synapse_new.prev_gradient = synapse_old.prev_gradient
    synapse_new.prev_delta = synapse_old.prev_delta
    synapse_new.prev_delta_w = synapse_old.prev_delta_w
    synapse_new.m_current = synapse_old.m_current
    synapse_new.v_current = synapse_old.v_current
    synapse_new.m_prev = synapse_old.m_prev
    synapse_new.v_prev = synapse_old.v_prev
    return synapse_new
  end

  def inspect
    pp @weight
    pp @source_neuron
    pp @dest_neuron
  end
end
