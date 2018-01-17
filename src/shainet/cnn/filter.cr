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
    end

    # Adds padding to all Filters of input data
    def _pad(filters : Array(Filter), padding : Int32)
      input_data = filters.clone # Array of filter class
      padded_data = Array(Array(Array(Neuron))).new

      if padding == 0
        input_data.each { |filter| padded_data << filter.neurons.clone }
      else
        blank_neuron = Neuron.new("memory")
        blank_neuron.activation = 0.0

        input_data.each do |filter|
          # Add padding at the sides
          filter.neurons.each do |row|
            padding.times { row << blank_neuron }
            padding.times { row.insert(0, blank_neuron) }
          end
          # Add padding at the top/bottom
          padding_row = Array(Neuron).new(filter.neurons.first.size) { blank_neuron }
          padding.times { filter.neurons << padding_row }
          padding.times { filter.neurons.insert(0, padding_row) }
          padded_data << filter.neurons
        end
      end
      return padded_data
    end

    # Adds padding to all channels of input data
    def _pad(filters : Array(Array(Array(Array(Neuron)))), padding : Int32)
      input_data = filters.first.clone # Array of all channels
      if padding == 0
        return input_data
      else
        blank_neuron = Neuron.new("memory")
        blank_neuron.activation = 0.0
        padded_data = input_data

        # Go over each channel and add padding
        padded_data.size.times do |channel|
          # Add padding at the sides
          padded_data[channel].each do |row|
            padding.times { row << blank_neuron }
            padding.times { row.insert(0, blank_neuron) }
          end
          # Add padding at the top/bottom
          padding_row = Array(Neuron).new(padded_data.first.first.size) { blank_neuron }
          padding.times { padded_data[channel] << padding_row }
          padding.times { padded_data[channel].insert(0, padding_row) }
        end
        return padded_data
      end
    end

    def propagate_forward(input_layer : ConvLayer | CNNLayer)
      padded_data = _pad(input_layer.filters, @padding) # Array of all channels or all filters

      # Starting locations
      input_x = input_y = output_x = output_y = 0

      # Takes a small window from the input data (Channel/Filter x Width x Height) to preform feed forward
      # Slides the window over the input data volume and updates each neuron of the filter
      # The window depth is the number of all channels/filters (depending on previous layer)
      while input_y < (padded_data.first.size - @window_size + @stride)         # Break out of y
        while input_x < (padded_data.first.first.size - @window_size + @stride) # Break out of x
          window = padded_data.map { |channel| channel[input_y..(input_y + @window_size - 1)].map { |row| row[input_x..(input_x + @window_size - 1)] } }
          target_neuron = @neurons[output_y][output_x]

          # Gather the weighted activations from the entire window
          input_sum = Float64.new(0)
          @synapses.size.times do |channel|
            @synapses[channel].size.times do |row|
              @synapses[channel][row].size.times do |col| # Synapses are CnnSynpase in this case
              # Save the weighted activations from previous layer
                input_sum += @synapses[channel][row][col].weight*window[channel][row][col].activation
              end
            end
          end

          # Add bias and apply activation function
          target_neuron.input_sum = input_sum + @bias
          target_neuron.activation, target_neuron.sigma_prime = @activation_function.call(target_neuron.input_sum)

          # Go to next window horizontaly
          input_x += @stride
          output_x += 1
        end
        # Go to next window verticaly
        input_x = output_x = 0
        input_y += @stride
        output_y += 1
      end
    end

    def propagate_backward(next_layer : ConvLayer)
      padded_data = _pad([self], next_layer.padding) # Array of all channels or all filters

      # Starting locations
      input_x = input_y = output_x = output_y = 0

      # Update the gradients of all neurons in current layer and weight gradients for the filters of the next layer
      next_layer.filters.size.times do |filter|
        # Takes a small window from the input data (Channel/Filter x Width x Height) to preform feed forward
        # Slides the window over the input data volume and updates each neuron of the filter
        # The window depth is the number of all channels/filters (depending on previous layer)
        while input_y < (padded_data.first.size - @window_size + @stride)         # Break out of y
          while input_x < (padded_data.first.first.size - @window_size + @stride) # Break out of x
            window = padded_data.map { |self_filter| self_filter[input_y..(input_y + @window_size - 1)].map { |row| row[input_x..(input_x + @window_size - 1)] } }
            source_neuron = next_layer.filters[filter].neurons[output_y][output_x]

            # update the weighted error for the entire window
            synapses = next_layer.filters[filter].synapses
            # input_sum = Float64.new(0)
            synapses.size.times do |channel|
              synapses[channel].size.times do |row|
                synapses[channel][row].size.times do |col| # Synapses are CnnSynpase in this case
                # Propagate the error from next layer to self, keeping in mind that all the filters in next layer contribute to the error
                  target_neuron = @neurons[row][col]
                  target_neuron.gradient_sum += synapses[channel][row][col].weight*source_neuron.gradient

                  # Save the error sum for updating the weights later
                  synapses[channel][row][col].gradient_sum += source_neuron.gradient*target_neuron.activation
                end
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

      # update gradients of all neurons in all filters
      @neurons.each do |row|
        row.each do |neuron|
          neuron.gradient = neuron.gradient_sum*neuron.sigma_prime
          neuron.gradient_sum = 0.0
        end
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
