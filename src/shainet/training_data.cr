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
      @yrange = 1
      @ymin = 0
    end

    def data
      arr = Array(Array(Array(Float64))).new
      @normalized_inputs.each_with_index do |i_arr, i|
        arr << [@normalized_inputs[i], @normalized_outputs[i]]
      end
      arr
    end

    def normalize_min_max
      # Get inputs min-max
      i_max = Array(GenNum).new
      i_min = Array(GenNum).new
      @inputs.transpose.each { |a| i_max << a.max }
      @inputs.transpose.each { |a| i_min << a.min }

      # Get outputs min-max
      o_max = Array(GenNum).new
      o_min = Array(GenNum).new
      @outputs.transpose.each { |a| o_max << a.max }
      @outputs.transpose.each { |a| o_min << a.min }

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
      xrange = xmax - xmin
      value = (@ymin + (x - xmin) * (@yrange.to_f64 / xrange)).to_f64
      value
    end
  end
end
