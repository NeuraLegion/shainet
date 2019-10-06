require "csv"

module SHAInet
  class Data
    @yrange : Int32
    @ymax : Int32
    @ymin : Int32

    property :normalized_outputs, :normalized_inputs, :labels, :outputs
    getter :inputs

    # @data_pairs :
    # Takes a path to a CSV file, a range of inputs and the index of the target column.
    # Returns a SHAInet::Data object.
    # ```
    # data = SHAInet::Data.new_with_csv_input_target("iris.csv", 0..3, 4)
    # ```
    def self.new_with_csv_input_target(csv_file_path, input_column_range, target_column)
      inputs = Array(Array(Float64)).new
      outputs = Array(Array(Float64)).new
      outputs_as_string = Array(String).new
      CSV.each_row(File.read(csv_file_path)) do |row|
        row_arr = Array(Float64).new
        row[input_column_range].each do |num|
          row_arr << num.to_f64
        end
        inputs << row_arr
        outputs_as_string << row[target_column]
      end
      d = Data.new(inputs, outputs)
      d.labels = outputs_as_string.uniq
      d.outputs = outputs_as_string.map { |string_output| d.array_for_label(string_output) }
      d.normalize_min_max
      d
    end

    def initialize(@inputs : Array(Array(Float64)), @outputs : Array(Array(Float64)))
      @ymax = 1
      @ymin = 0
      @yrange = @ymax - @ymin

      @i_min = Array(GenNum).new
      @i_max = Array(GenNum).new
      @o_min = Array(GenNum).new
      @o_max = Array(GenNum).new

      @labels = Array(String).new # Array of possible data labels
      @logger = Logger.new(STDOUT)

      @normalized_inputs = Array(Array(Float64)).new
      @normalized_outputs = Array(Array(Float64)).new
    end

    def data
      arr = Array(Array(Array(Float64))).new
      @normalized_inputs.each_with_index do |_, i|
        arr << [@normalized_inputs[i], @normalized_outputs[i]]
      end
      arr
    end

    def raw_data
      arr = Array(Array(Array(Float64))).new
      @inputs.each_with_index do |_, i|
        arr << [@inputs[i], @outputs[i]]
      end
      arr
    end

    def normalize_min_max(data : Array(Array(Float64)))
      normalized_data = Array(Array(Float64)).new

      # Get min-max
      data.transpose.each { |a| @i_max << a.max; @i_min << a.min }

      data.each do |row|
        normalized_data << normalize_inputs(row)
      end

      normalized_data
    end

    def normalize_min_max
      # Get inputs min-max
      @inputs.transpose.each { |a| @i_max << a.max; @i_min << a.min }

      # Get outputs min-max
      @outputs.transpose.each { |a| @o_max << a.max; @o_min << a.min }

      @inputs.each do |row|
        @normalized_inputs << normalize_inputs(row)
      end

      @outputs.each do |row|
        @normalized_outputs << normalize_outputs(row)
      end
    end

    def normalize_inputs(inputs : Array(GenNum))
      results = Array(Float64).new
      inputs.each_with_index do |input, i|
        results << normalize(input, @i_min[i], @i_max[i])
      end
      results
    end

    def normalize_outputs(outputs : Array(GenNum))
      results = Array(Float64).new
      outputs.each_with_index do |output, i|
        results << normalize(output, @o_min[i], @o_max[i])
      end
      results
    end

    def normalize(x, xmin, xmax)
      range = xmax - xmin
      adj_x = x.to_f64 - (xmin + @ymin)
      norm = (@yrange / range)
      value = adj_x * norm
      return 0.0 if value.nan?
      value
    end

    def denormalize_outputs(outputs : Array(GenNum))
      results = Array(Float64).new
      outputs.each_with_index do |output, i|
        results << denormalize(output, @o_min[i], @o_max[i])
      end
      results
    end

    def denormalize(x, xmin, xmax)
      range = xmax - xmin
      denorm = x.to_f64 * (range / @yrange)
      adj_x = @ymin + xmin
      value = denorm + adj_x
      return 0.0 if value.nan?
      value
    end

    # Splits the receiver in a TrainingData and a TestData object according to factor
    def split(factor)
      training_set_size = (data.size * factor).to_i

      # puts data
      shuffled_data = data.shuffle

      # puts shuffled_data

      training_set = shuffled_data[0..training_set_size - 1]
      test_set = shuffled_data[training_set_size..shuffled_data.size - 1]

      # puts training_set

      @logger.info "Selected #{training_set.size} / #{data.size} rows for training"
      training_data = SHAInet::TrainingData.new(training_set.map { |el| el[0] }, training_set.map { |el| el[1] })
      training_data.labels = @labels
      training_data.normalize_min_max

      @logger.info "Selected #{test_set.size} / #{data.size} rows for testing"
      test_data = SHAInet::TestData.new(test_set.map { |el| el[0] }, test_set.map { |el| el[1] })
      test_data.labels = @labels
      test_data.normalize_min_max

      return training_data, test_data
    end

    # Receives an array of labels (String or Symbol) and sets them for this Data object
    def labels=(label_array)
      @labels = label_array.map(&.to_s)
      @logger.info("Labels are #{@labels.join(", ")}") if self.class.name == "SHAInet::Data"
    end

    # Takes a label as a String and returns the corresponding output array
    def array_for_label(a_label)
      @labels.map { |label| a_label == label ? 1.to_f64 : 0.to_f64 }
    end

    # Takes an output array of 0,1s and returns the corresponding label
    def label_for_array(an_array)
      index = an_array.index(an_array.max.to_f64)
      index ? @labels[index] : ""
    end

    def size
      @inputs.size
    end

    def to_onehot(data : Array(Array(Float64)), vector_size : Int32)
      data.each_with_index do |point, i|
        lbl = point.first.clone.to_i
        one_hot = Array(Float64).new(vector_size) { 0.0 }
        one_hot[lbl] = 1.0
        data[i] = one_hot
      end

      data
    end
  end
end
