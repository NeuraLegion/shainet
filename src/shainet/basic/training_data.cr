module SHAInet
  class TrainingData
    getter data_pairs : Array(Array(Array(Array(Array(Float64))) | Array(Float64)))
    @normalized_inputs : Array(Array(Float64))
    @normalized_outputs : Array(Array(Float64))
    @yrange : Int32
    @ymin : Int32

    getter :normalized_outputs, :normalized_inputs

    # For standard NN
    def initialize(@inputs : Array(Array(Float64)), @outputs : Array(Array(Float64)))
      @normalized_inputs = Array(Array(Float64)).new
      @normalized_outputs = Array(Array(Float64)).new
      @ymax = 1
      @ymin = 0
      @yrange = @ymax - @ymin

      #
      @data = Array(Array(Float64)).new
      @data_pairs = Array(Array(Array(Array(Array(Float64))) | Array(Float64))).new
    end

    # For MNIST test using CNN
    def initialize(@data : Array(Array(Float64)))
      @data_pairs = Array(Array(Array(Array(Array(Float64))) | Array(Float64))).new

      #
      @inputs = Array(Array(Float64)).new
      @outputs = Array(Array(Float64)).new
      @normalized_inputs = Array(Array(Float64)).new
      @normalized_outputs = Array(Array(Float64)).new
      @ymax = 1
      @ymin = 0
      @yrange = @ymax - @ymin
    end

    # For MNIST using CNN
    def for_mnist_conv
      @data.each do |sample|
        pair = Array(Array(Array(Array(Float64))) | Array(Float64)).new

        output = Array(Float64).new(10) { 0.0 } # One-hot vector output (for 10 digits)
        output[sample.first.to_i] = 1.0

        input = Array(Array(Array(Float64))).new # Input may have multiple channels (not in MNIST case though)
        channel = Array(Array(Float64)).new
        sample[1..-1].each_slice(28) do |row| # Here we have only one channel since its MNIST
          row.each { |value| value.to_f64 }
          channel << row
        end
        input << channel

        pair << input
        pair << output
        @data_pairs << pair
      end
    end

    def data
      arr = Array(Array(Array(Float64))).new
      @normalized_inputs.each_with_index do |i_arr, i|
        arr << [@normalized_inputs[i], @normalized_outputs[i]]
      end
      arr
    end

    def raw_data
      arr = Array(Array(Array(Float64))).new
      @inputs.each_with_index do |i_arr, i|
        arr << [@inputs[i], @outputs[i]]
      end
      arr
    end

    def normalize_min_max
      # Get inputs min-max
      i_max = Array(GenNum).new
      i_min = Array(GenNum).new
      @inputs.transpose.each { |a| i_max << a.max; i_min << a.min }

      # Get outputs min-max
      o_max = Array(GenNum).new
      o_min = Array(GenNum).new
      @outputs.transpose.each { |a| o_max << a.max; o_min << a.min }

      @inputs.each do |row|
        row_array = Array(Float64).new
        row.each_with_index do |member, i|
          row_array << normalize(member, i_min[i], i_max[i])
        end
        @normalized_inputs << row_array
      end

      @outputs.each do |row|
        row_array = Array(Float64).new
        row.each_with_index do |member, i|
          row_array << normalize(member, o_min[i], o_max[i])
        end
        @normalized_outputs << row_array
      end
    end

    def normalize(x : GenNum, xmin : GenNum, xmax : GenNum) : Float64
      value = (@ymin + (x - xmin) * (@yrange.to_f64 / (xmax - xmin))).to_f64
      return 0.0 if value.nan?
      value
    end
  end
end
