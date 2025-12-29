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
    @gpu_in_ws : Array(CudaMatrix) = [] of CudaMatrix
    @gpu_out_ws : Array(CudaMatrix) = [] of CudaMatrix
    @ws_batch_size : Int32 = 0

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
        json = JSON.parse(line)

        input_json : JSON::Any
        output_json : JSON::Any

        if json.raw.is_a?(Array)
          pair = json.as_a
          next if pair.size < 2
          input_json = pair[0]
          output_json = pair[1]
        elsif json.raw.is_a?(Hash)
          obj = json.as_h
          if obj["input"]?
            input_json = obj["input"].not_nil!
          elsif obj["inputs"]?
            input_json = obj["inputs"].not_nil!
          else
            next
          end

          if obj["target"]?
            output_json = obj["target"].not_nil!
          elsif obj["output"]?
            output_json = obj["output"].not_nil!
          else
            next
          end
        else
          next
        end

        input = parse_array(input_json)
        output = parse_array(output_json)
        batch << [input, output]
      end

      return batch unless @gpu_batches && CUDA.fully_available?

      # Determine matrix dimensions from first sample
      if batch.empty?
        return [] of Array(CudaMatrix)
      end

      first_in = batch.first[0]
      first_out = batch.first[1]

      get_dims = ->(d : Datum) do
        if d.is_a?(Array(Array(Float64)))
          {d.size, d.first.size}
        else
          {1, d.as(Array(Float64)).size}
        end
      end

      in_rows, in_cols = get_dims.call(first_in)
      out_rows, out_cols = get_dims.call(first_out)

      # Reallocate persistent buffers when batch size or dimensions change
      if @gpu_in_ws.empty? || @ws_batch_size != batch_size ||
         @gpu_in_ws.first.rows != in_rows || @gpu_in_ws.first.cols != in_cols ||
         @gpu_out_ws.first.rows != out_rows || @gpu_out_ws.first.cols != out_cols
        @gpu_in_ws = Array(CudaMatrix).new(batch_size) { CudaMatrix.new(in_rows, in_cols) }
        @gpu_out_ws = Array(CudaMatrix).new(batch_size) { CudaMatrix.new(out_rows, out_cols) }
        @ws_batch_size = batch_size
      end

      gpu_batch = [] of Array(CudaMatrix)

      batch.each_with_index do |ex, idx|
        inp = ex[0]
        out_m = ex[1]

        GPUMemory.to_gpu!(inp, @gpu_in_ws[idx])
        GPUMemory.to_gpu!(out_m, @gpu_out_ws[idx])

        gpu_batch << [@gpu_in_ws[idx], @gpu_out_ws[idx]]
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
      while line = @file.gets
        @buffer << line
        count += 1
        break if count >= @chunk_size
      end
      shuffle! if @shuffle
    end

    # Parses a JSON array and converts all numeric values to Float64. Supports
    # both 1‑D and 2‑D arrays which is useful for tokenized inputs.
    private def parse_array(json_any : JSON::Any) : Datum
      if json_any.raw.is_a?(Array)
        arr = json_any.as_a
        return [] of Float64 if arr.empty?
        if arr.first.raw.is_a?(Array)
          Array(Array(Float64)).from_json(json_any.to_json)
        else
          Array(Float64).from_json(json_any.to_json)
        end
      else
        [json_any.as_f] of Float64
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
