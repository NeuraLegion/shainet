module SHAInet
  # StreamingData reads training data lazily from a text file. Each line should
  # contain a JSON array: [[inputs...], [outputs...]]
  class StreamingData
    @path : String
    @file : File

    def initialize(@path : String)
      @file = File.new(@path, "r")
    end

    def next_batch(batch_size : Int32)
      batch = [] of Array(Array(Float64))
      batch_size.times do
        line = @file.gets
        break unless line
        pair = JSON.parse(line).as_a
        input = pair[0].as_a.map { |x| x.as_f }
        output = pair[1].as_a.map { |x| x.as_f }
        batch << [input, output]
      end
      batch
    end

    def rewind
      @file.close
      @file = File.new(@path, "r")
    end
  end
end
