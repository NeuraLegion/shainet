require "json"

module SHAInet
  # Reads tensors from HuggingFace SafeTensors files without any Python dependency.
  # Format: [8-byte u64 LE header_size][JSON header][raw tensor data]
  module SafeTensors
    enum DType
      F16
      BF16
      F32
      F64
      I8
      I16
      I32
      I64
      U8
      U16
      U32
      U64
      BOOL

      def byte_size : Int32
        case self
        when F16, BF16, I16, U16 then 2
        when F32, I32, U32       then 4
        when F64, I64, U64       then 8
        when I8, U8, BOOL       then 1
        else                          raise "Unknown dtype byte size"
        end
      end
    end

    record TensorInfo, dtype : DType, shape : Array(Int64), data_offset_start : Int64, data_offset_end : Int64

    class File
      getter tensors : Hash(String, TensorInfo)
      getter metadata : Hash(String, String)?
      @io : ::IO
      @data_offset : Int64 # byte position where tensor data begins

      def initialize(path : String)
        @io = ::File.open(path, "rb")
        @metadata = nil

        # Read header size (first 8 bytes, u64 LE)
        header_size_bytes = Bytes.new(8)
        @io.read_fully(header_size_bytes)
        header_size = IO::ByteFormat::LittleEndian.decode(UInt64, header_size_bytes)

        # Guard against corrupt/malicious files (100MB header limit)
        raise "SafeTensors header too large: #{header_size} bytes" if header_size > 100_000_000_u64

        # Read JSON header
        header_bytes = Bytes.new(header_size.to_i32)
        @io.read_fully(header_bytes)
        header_json = JSON.parse(String.new(header_bytes))

        @data_offset = 8_i64 + header_size.to_i64
        @tensors = Hash(String, TensorInfo).new

        header_json.as_h.each do |key, value|
          if key == "__metadata__"
            @metadata = value.as_h.transform_values(&.as_s)
            next
          end

          obj = value.as_h
          dtype = DType.parse(obj["dtype"].as_s)
          shape = obj["shape"].as_a.map(&.as_i64)
          offsets = obj["data_offsets"].as_a
          start_offset = offsets[0].as_i64
          end_offset = offsets[1].as_i64

          @tensors[key] = TensorInfo.new(dtype, shape, start_offset, end_offset)
        end
      end

      def close
        @io.close
      end

      def tensor_names : Array(String)
        @tensors.keys
      end

      def has_tensor?(name : String) : Bool
        @tensors.has_key?(name)
      end

      # Read a tensor as Float32 array (handles F32 and F16 conversion)
      def read_f32(name : String) : Array(Float32)
        info = @tensors[name]? || raise "Tensor '#{name}' not found"
        byte_count = (info.data_offset_end - info.data_offset_start).to_i32

        @io.seek(@data_offset + info.data_offset_start)
        raw = Bytes.new(byte_count)
        @io.read_fully(raw)

        case info.dtype
        when .f32?
          count = byte_count // 4
          Array(Float32).new(count) do |i|
            IO::ByteFormat::LittleEndian.decode(Float32, raw[i * 4, 4])
          end
        when .f16?
          count = byte_count // 2
          Array(Float32).new(count) do |i|
            f16_to_f32(IO::ByteFormat::LittleEndian.decode(UInt16, raw[i * 2, 2]))
          end
        when .bf16?
          count = byte_count // 2
          Array(Float32).new(count) do |i|
            bf16_to_f32(IO::ByteFormat::LittleEndian.decode(UInt16, raw[i * 2, 2]))
          end
        when .f64?
          count = byte_count // 8
          Array(Float32).new(count) do |i|
            IO::ByteFormat::LittleEndian.decode(Float64, raw[i * 8, 8]).to_f32
          end
        else
          raise "Unsupported dtype #{info.dtype} for float conversion"
        end
      end

      # Read a tensor as Float64 array
      def read_f64(name : String) : Array(Float64)
        info = @tensors[name]? || raise "Tensor '#{name}' not found"
        byte_count = (info.data_offset_end - info.data_offset_start).to_i32

        @io.seek(@data_offset + info.data_offset_start)
        raw = Bytes.new(byte_count)
        @io.read_fully(raw)

        case info.dtype
        when .f32?
          count = byte_count // 4
          Array(Float64).new(count) do |i|
            IO::ByteFormat::LittleEndian.decode(Float32, raw[i * 4, 4]).to_f64
          end
        when .f64?
          count = byte_count // 8
          Array(Float64).new(count) do |i|
            IO::ByteFormat::LittleEndian.decode(Float64, raw[i * 8, 8])
          end
        when .f16?
          count = byte_count // 2
          Array(Float64).new(count) do |i|
            f16_to_f32(IO::ByteFormat::LittleEndian.decode(UInt16, raw[i * 2, 2])).to_f64
          end
        when .bf16?
          count = byte_count // 2
          Array(Float64).new(count) do |i|
            bf16_to_f32(IO::ByteFormat::LittleEndian.decode(UInt16, raw[i * 2, 2])).to_f64
          end
        else
          raise "Unsupported dtype #{info.dtype} for float conversion"
        end
      end

      # Read tensor into a 2D SimpleMatrix (row-major)
      def read_matrix(name : String) : SimpleMatrix
        info = @tensors[name]? || raise "Tensor '#{name}' not found"
        shape = info.shape

        rows = shape.size == 1 ? 1 : shape[0].to_i32
        cols = shape.size == 1 ? shape[0].to_i32 : shape[1].to_i32
        raise "Cannot read tensor '#{name}' with #{shape.size}D shape as matrix" if shape.size > 2

        m = SimpleMatrix.new(rows, cols)
        count = rows * cols
        byte_count = (info.data_offset_end - info.data_offset_start).to_i32
        @io.seek(@data_offset + info.data_offset_start)

        if info.dtype.f32?
          # Fast path: raw memcpy (F32 LE on disk → F32 LE in memory)
          expected = count * 4
          raise "SafeTensors: tensor '#{name}' byte_count #{byte_count} != expected #{expected}" if byte_count != expected
          raw = Bytes.new(byte_count)
          @io.read_fully(raw)
          raw.to_unsafe.copy_to(m.data.to_unsafe.as(Pointer(UInt8)), byte_count)
        else
          # Slow path: convert from other dtypes
          data = read_f32(name)
          data.size.times { |i| m.data[i] = data[i] }
        end
        m
      end

      private def f16_to_f32(bits : UInt16) : Float32
        sign = (bits >> 15).to_u32
        exp = ((bits >> 10) & 0x1F).to_u32
        mant = (bits & 0x3FF).to_u32

        if exp == 0
          # Subnormal or zero
          if mant == 0
            f32_bits = sign << 31
          else
            # Subnormal: normalize
            e = -1
            m = mant
            while (m & 0x400) == 0
              m <<= 1
              e -= 1
            end
            m &= 0x3FF
            f32_bits = (sign << 31) | ((127 + e).to_u32 << 23) | (m.to_u32 << 13)
          end
        elsif exp == 31
          # Inf or NaN
          f32_bits = (sign << 31) | (0xFF_u32 << 23) | (mant.to_u32 << 13)
        else
          # Normal
          f32_bits = (sign << 31) | ((exp + 112).to_u32 << 23) | (mant.to_u32 << 13)
        end

        pointerof(f32_bits).as(Pointer(Float32)).value
      end

      private def bf16_to_f32(bits : UInt16) : Float32
        # BF16 is just the upper 16 bits of F32
        f32_bits = bits.to_u32 << 16
        pointerof(f32_bits).as(Pointer(Float32)).value
      end
    end
  end
end
