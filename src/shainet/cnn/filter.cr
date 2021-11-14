module SHAInet
  class Filter
    getter input_surface : Array(Int32), window_size : Int32, stride : Int32, padding : Int32, activation_function : ActivationFunction
    property neurons : Array(Array(Neuron)), synapses : Array(Array(Array(CnnSynapse)))
    property bias : Float64, prev_bias : Float64, bias_grad : Float64, bias_grad_sum : Float64, bias_grad_batch : Float64
    property prev_bias_grad : Float64, prev_delta : Float64, prev_delta_b : Float64
    property m_current : Float64, v_current : Float64, m_prev : Float64, v_prev : Float64

    def initialize(@input_surface : Array(Int32), # expecting [width, height, channels]
                   @padding : Int32 = 0,
                   @window_size : Int32 = 1,
                   @stride : Int32 = 1,
                   @activation_function : ActivationFunction = SHAInet.none)
      #
      @neurons = Array(Array(Neuron)).new(input_surface[1]) {
        Array(Neuron).new(input_surface[0]) { Neuron.new("memory") }
      }

      @synapses = Array(Array(Array(CnnSynapse))).new(input_surface[2]) {
        Array(Array(CnnSynapse)).new(@window_size) {
          Array(CnnSynapse).new(@window_size) { CnnSynapse.new }
        }
      }

      @blank_neuron = Neuron.new("memory") # This is needed for padding
      @blank_neuron.activation = 0.0
      @blank_neuron.gradient = 0.0

      @bias = rand(-0.1..0.1).to_f64
      @prev_bias = rand(-0.1..0.1).to_f64 # Needed for delta rule improvement using momentum
      @bias_grad = Float64.new(0)
      @bias_grad_sum = Float64.new(0)   # For conv-layer backprop
      @bias_grad_batch = Float64.new(0) # For mini-batch backprop

      # Parameters needed for Rprop
      @prev_bias_grad = rand(-0.1..0.1).to_f64
      @prev_delta = rand(0.0..0.1).to_f64
      @prev_delta_b = rand(-0.1..0.1).to_f64

      # Parameters needed for Adam
      @m_current = Float64.new(0) # Current moment value
      @v_current = Float64.new(0) # Current moment**2 value
      @m_prev = Float64.new(0)    # Previous moment value
      @v_prev = Float64.new(0)    # Previous moment**2 value

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

    def propagate_backward(input_layer : ConvLayer | CNNLayer, batch : Bool = false)
      # Starting locations
      input_x = input_y = -@padding
      output_x = output_y = 0

      # Similar to forward propogation, only in reverse
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
                @bias_grad_sum += source_neuron.gradient

                if batch == true
                  target_neuron.gradient_batch += target_neuron.gradient
                  synapse.gradient_batch += target_neuron.activation*source_neuron.gradient
                  @bias_grad_batch += source_neuron.gradient
                end
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
