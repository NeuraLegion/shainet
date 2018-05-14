require "./data"

module SHAInet
  class CNNData < Data
    alias CNNinputData = Array(Array(Array(Array(Float64)))) # Array of all 3D inputs
    alias CNNoutputData = Array(Array(Float64))              # Array of all expected outputs
    alias CNNPair = {input: Array(Array(Array(Float64))), output: Array(Float64)}

    @yrange : Int32
    @ymin : Int32

    getter :normalized_outputs, :normalized_inputs, :labels, :data_pairs
    setter :outputs

    # When inputs are one-dimentional
    def initialize(@inputs : Array(Array(Float64)), @outputs : Array(Array(Float64)))
      @normalized_inputs = Array(Array(Float64)).new
      @normalized_outputs = Array(Array(Float64)).new
      @ymax = 1
      @ymin = 0
      @yrange = @ymax - @ymin
      @labels = Array(String).new # Array of possible data labels
      @logger = Logger.new(STDOUT)

      @data_pairs = Array({input: Array(Array(Array(Float64))), output: Array(Float64)}).new
    end

    # When inputs are three-dimentional
    def initialize(@inputs : CNNinputData, @outputs : CNNoutputData)
      @normalized_inputs = Array(Array(Array(Array(Float64)))).new
      @normalized_outputs = Array(Array(Float64)).new
      @ymax = 1
      @ymin = 0
      @yrange = @ymax - @ymin
      @labels = Array(String).new # Array of possible data labels
      @logger = Logger.new(STDOUT)

      @data_pairs = Array(CNNPair).new
    end

    # Normalize input to 3D image and change expected outputs to 1-hot vector
    def for_mnist_conv
      @inputs.each_with_index do |raw_input, i|
        raw_input.each_with_index { |v, i| raw_input[i] = normalize(x: v.as(Float64), xmin: 0, xmax: 255) }
        channel = vector_to_2d(vector: raw_input, window_size: 28)

        normalized_input = [channel] # Mnist has only one channel

        normalized_output = Array(Float64).new(10) { 0.0 } # One-hot vector output (for 10 digits)
        normalized_output[@outputs[i].first.to_i] = 1.0

        @data_pairs.as(Array(CNNPair)) << {input: normalized_input, output: normalized_output}
      end
    end

    def vector_to_2d(vector : Array(Float64), window_size : Int32)
      channel = Array(Array(Float64)).new
      vector.each_slice(window_size) { |row| channel << row }
      return channel
    end
  end
end
