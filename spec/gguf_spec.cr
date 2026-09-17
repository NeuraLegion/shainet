require "./spec_helper"

# Build a minimal valid GGUF v3 file in memory and verify the parser reads it back.
private def write_gguf_string(io : IO, s : String)
  io.write_bytes(s.bytesize.to_u64, IO::ByteFormat::LittleEndian)
  io.write(s.to_slice)
end

private def build_test_gguf(path : String)
  File.open(path, "w") do |f|
    # Header: magic, version, tensor_count, kv_count
    f.write_bytes(0x46554747_u32, IO::ByteFormat::LittleEndian) # GGUF
    f.write_bytes(3_u32, IO::ByteFormat::LittleEndian)          # v3
    f.write_bytes(1_u64, IO::ByteFormat::LittleEndian)          # 1 tensor
    f.write_bytes(3_u64, IO::ByteFormat::LittleEndian)          # 3 KV pairs

    # KV 1: general.architecture = "llama" (string)
    write_gguf_string(f, "general.architecture")
    f.write_bytes(8_u32, IO::ByteFormat::LittleEndian) # STRING
    write_gguf_string(f, "llama")

    # KV 2: llama.block_count = 2 (uint32)
    write_gguf_string(f, "llama.block_count")
    f.write_bytes(4_u32, IO::ByteFormat::LittleEndian) # UINT32
    f.write_bytes(2_u32, IO::ByteFormat::LittleEndian)

    # KV 3: general.name = "test" (string)
    write_gguf_string(f, "general.name")
    f.write_bytes(8_u32, IO::ByteFormat::LittleEndian) # STRING
    write_gguf_string(f, "test")

    # Tensor info: "weight" shape [4, 2], type F32, offset 0
    write_gguf_string(f, "weight")
    f.write_bytes(2_u32, IO::ByteFormat::LittleEndian) # ndim
    f.write_bytes(4_u64, IO::ByteFormat::LittleEndian) # dim 0
    f.write_bytes(2_u64, IO::ByteFormat::LittleEndian) # dim 1
    f.write_bytes(0_u32, IO::ByteFormat::LittleEndian) # type F32
    f.write_bytes(0_u64, IO::ByteFormat::LittleEndian) # offset

    # Pad to alignment (32 bytes)
    pos = f.pos
    pad = (32 - (pos % 32)) % 32
    pad.to_i32.times { f.write_byte(0_u8) }

    # Tensor data: 8 float32 values
    8.times { |i| f.write_bytes((i + 1).to_f32, IO::ByteFormat::LittleEndian) }
  end
end

describe SHAInet::GGUF do
  test_path = File.tempname("test", ".gguf")

  before_each { build_test_gguf(test_path) }
  after_each { File.delete(test_path) if File.exists?(test_path) }

  it "parses the header and metadata" do
    gf = SHAInet::GGUF::File.open(test_path)
    gf.version.should eq 3_u32
    gf.metadata.size.should eq 3
    gf.meta_string("general.architecture").should eq "llama"
    gf.meta_u32("llama.block_count").should eq 2_u32
    gf.meta_string("general.name").should eq "test"
    gf.close
  end

  it "reads tensor info" do
    gf = SHAInet::GGUF::File.open(test_path)
    gf.tensors.size.should eq 1
    ti = gf.tensors["weight"]
    ti.shape.should eq [4_u64, 2_u64]
    ti.type.should eq SHAInet::GGUF::GGMLType::F32
    ti.element_count.should eq 8_u64
    ti.byte_size.should eq 32_u64 # 8 * 4
    gf.close
  end

  it "reads tensor data" do
    gf = SHAInet::GGUF::File.open(test_path)
    ti = gf.tensors["weight"]
    buf = Bytes.new(ti.byte_size.to_i32)
    gf.read_tensor_data(ti, buf)
    # First value should be 1.0f32
    v = IO::ByteFormat::LittleEndian.decode(Float32, buf[0, 4])
    v.should be_close(1.0, 1e-6)
    # Last value should be 8.0f32
    v = IO::ByteFormat::LittleEndian.decode(Float32, buf[28, 4])
    v.should be_close(8.0, 1e-6)
    gf.close
  end

  it "computes Q4_K byte size correctly" do
    # 256 values per block, 144 bytes per block
    info = SHAInet::GGUF::TensorInfo.new("test", [512_u64, 256_u64], SHAInet::GGUF::GGMLType::Q4_K, 0_u64)
    # 131072 elements / 256 per block = 512 blocks * 144 = 73728
    info.byte_size.should eq 73728_u64
  end

  it "computes Q6_K byte size correctly" do
    info = SHAInet::GGUF::TensorInfo.new("test", [256_u64], SHAInet::GGUF::GGMLType::Q6_K, 0_u64)
    # 256 / 256 = 1 block * 210 = 210
    info.byte_size.should eq 210_u64
  end
end
