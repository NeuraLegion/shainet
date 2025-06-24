module SHAInet
  # StreamingData reads training data lazily from a text file. Each line should
  # contain a JSON array: [[inputs...], [outputs...]]
  class StreamingData
    alias Datum = Array(Float64) | Array(Array(Float64))
    @path : String
    @lines : Array(String)
    @index : Int32 = 0
    @shuffle : Bool

    # Reads data from `path`. When `shuffle` is true the file is read into
    # memory and shuffled at the beginning of every epoch.
    def initialize(@path : String, @shuffle : Bool = false)
      @lines = File.read_lines(@path)
      shuffle! if @shuffle
    end

    # Returns the next `batch_size` examples. Each line may contain either a
    # flat array of numbers or nested arrays of token ids. All numbers are
    # converted to `Float64`.
    def next_batch(batch_size : Int32)
      batch = [] of Array(Datum)
      batch_size.times do
        break if @index >= @lines.size
        pair = JSON.parse(@lines[@index]).as_a
        @index += 1
        input = parse_array(pair[0])
        output = parse_array(pair[1])
        batch << [input, output]
      end
      batch
    end

    # Resets the data pointer for a new epoch and reshuffles if enabled.
    def rewind
      @index = 0
      shuffle! if @shuffle
    end

    private def shuffle!
      @lines.shuffle!
    end

    # Parses a JSON array and converts all numeric values to Float64. Supports
    # both 1‑D and 2‑D arrays which is useful for tokenized inputs.
    private def parse_array(json_any : JSON::Any) : Datum
      arr = json_any.as_a
      return [] of Float64 if arr.empty?
      if arr.first.raw.is_a?(Array)
        Array(Array(Float64)).from_json(json_any.to_json)
      else
        Array(Float64).from_json(json_any.to_json)
      end
    end
  end
end
