module SHAInet
  # StreamingData reads training data lazily from a text file. Each line should
  # contain a JSON array: [[inputs...], [outputs...]]
  class StreamingData
    alias Datum = Array(Float64) | Array(Array(Float64))
    @path : String
    @file : File
    @buffer : Array(String)
    @index : Int32 = 0
    @shuffle : Bool
    @chunk_size : Int32
    @gpu_batches : Bool
    getter gpu_batches

    # Reads data from `path`. The file is buffered in chunks to avoid loading
    # everything into memory. When `shuffle` is true the buffer is shuffled at
    # the beginning of each chunk.
    def initialize(@path : String, @shuffle : Bool = false, @chunk_size : Int32 = 1024, gpu_batches : Bool = false)
      @gpu_batches = gpu_batches
      @file = File.new(@path)
      @buffer = [] of String
      @index = 0
      read_chunk
    end

    # Returns the next `batch_size` examples. Each line may contain either a
    # flat array of numbers or nested arrays of token ids. All numbers are
    # converted to `Float64`.
    def next_batch(batch_size : Int32)
      batch = [] of Array(Datum)
      batch_size.times do
        line = next_line
        break unless line
        pair = JSON.parse(line).as_a
        input = parse_array(pair[0])
        output = parse_array(pair[1])
        batch << [input, output]
      end

      return batch unless @gpu_batches && CUDA.fully_available?

      gpu_batch = [] of Array(CudaMatrix)

      batch.each do |ex|
        inp = to_matrix(ex[0])
        out_m = to_matrix(ex[1])

        in_ws = CudaMatrix.get_workspace(inp.rows, inp.cols, "stream_in")
        out_ws = CudaMatrix.get_workspace(out_m.rows, out_m.cols, "stream_out")

        GPUMemory.to_gpu!(inp, in_ws)
        GPUMemory.to_gpu!(out_m, out_ws)

        gpu_batch << [in_ws, out_ws]
      end

      gpu_batch
    end

    # Similar to `next_batch` but returns GPU matrices when CUDA is available.
    def next_batch_gpu(batch_size : Int32)
      return next_batch(batch_size) unless CUDA.fully_available?

      prev = @gpu_batches
      @gpu_batches = true
      batch = next_batch(batch_size)
      @gpu_batches = prev
      batch.as(Array(Array(SimpleMatrix)))
    end

    # Resets the data pointer for a new epoch and reshuffles if enabled.
    def rewind
      @file.seek(0)
      read_chunk
    end

    private def shuffle!
      @buffer.shuffle!
    end

    private def next_line : String?
      if @index >= @buffer.size
        read_chunk
        return nil if @buffer.empty?
      end
      line = @buffer[@index]
      @index += 1
      line
    end

    private def read_chunk
      @buffer.clear
      @index = 0
      count = 0
      while (line = @file.gets)
        @buffer << line
        count += 1
        break if count >= @chunk_size
      end
      shuffle! if @shuffle
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

    private def to_matrix(d : Datum) : SimpleMatrix
      if d.is_a?(Array(Array(Float64)))
        SimpleMatrix.from_a(d)
      else
        SimpleMatrix.from_a([d.as(Array(Float64))])
      end
    end
  end
end
