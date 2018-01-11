module SHAInet
  class TrainingData
    @normalized_inputs : Array(Array(Float64))
    @normalized_outputs : Array(Array(Float64))
    @yrange : Int32
    @ymin : Int32

    getter :normalized_outputs, :normalized_inputs

    def initialize(@inputs : Array(Array(Float64)), @outputs : Array(Array(Float64)))
      @normalized_inputs = Array(Array(Float64)).new
      @normalized_outputs = Array(Array(Float64)).new
      @ymax = 1
      @ymin = 0
      @yrange = @ymax - @ymin
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
