require "./spec_helper"
require "json"
require "file_utils"

# Write a minimal single-file safetensors with the given 1-D F32 tensors.
# Format: [8-byte u64 LE header size][JSON header][raw little-endian F32 data].
private def write_safetensors(path : String, tensors : Hash(String, Array(Float32)))
  data = IO::Memory.new
  offsets = {} of String => Tuple(Int64, Int64)
  tensors.each do |name, vals|
    start = data.size.to_i64
    vals.each { |v| data.write_bytes(v, IO::ByteFormat::LittleEndian) }
    offsets[name] = {start, data.size.to_i64}
  end

  header = JSON.build do |j|
    j.object do
      tensors.each do |name, vals|
        j.field(name) do
          j.object do
            j.field("dtype", "F32")
            j.field("shape") { j.array { j.number(vals.size) } }
            j.field("data_offsets") { j.array { off = offsets[name]; j.number(off[0]); j.number(off[1]) } }
          end
        end
      end
    end
  end

  File.open(path, "wb") do |f|
    hbytes = header.to_slice
    f.write_bytes(hbytes.size.to_u64, IO::ByteFormat::LittleEndian)
    f.write(hbytes)
    f.write(data.to_slice)
  end
end

describe SHAInet::SafeTensors::ShardedFile do
  it "routes each tensor read to the shard that owns it" do
    dir = File.tempname("shards")
    Dir.mkdir_p(dir)
    begin
      write_safetensors(File.join(dir, "model-00001-of-00002.safetensors"), {"a" => [1.0_f32, 2.0_f32, 3.0_f32]})
      write_safetensors(File.join(dir, "model-00002-of-00002.safetensors"), {"b" => [4.0_f32, 5.0_f32]})
      index = {
        "metadata"   => {"total_size" => 20},
        "weight_map" => {
          "a" => "model-00001-of-00002.safetensors",
          "b" => "model-00002-of-00002.safetensors",
        },
      }
      idx_path = File.join(dir, "model.safetensors.index.json")
      File.write(idx_path, index.to_json)

      sf = SHAInet::SafeTensors::ShardedFile.new(dir, idx_path)
      begin
        sf.has_tensor?("a").should be_true
        sf.has_tensor?("b").should be_true
        sf.has_tensor?("missing").should be_false
        sf.tensor_names.sort.should eq(["a", "b"])

        sf.read_f32("a").should eq([1.0_f32, 2.0_f32, 3.0_f32])
        sf.read_f32("b").should eq([4.0_f32, 5.0_f32])

        ma = sf.read_matrix("a")
        ma.rows.should eq(1)
        ma.cols.should eq(3)
        ma[0, 2].should eq(3.0_f32)
      ensure
        sf.close
      end
    ensure
      FileUtils.rm_rf(dir)
    end
  end
end

describe "SHAInet::HFLoader.open_safetensors" do
  it "selects a single-file File when model.safetensors exists" do
    dir = File.tempname("single")
    Dir.mkdir_p(dir)
    begin
      write_safetensors(File.join(dir, "model.safetensors"), {"w" => [7.0_f32]})
      sf = SHAInet::HFLoader.open_safetensors(dir)
      begin
        sf.should be_a(SHAInet::SafeTensors::File)
        sf.read_f32("w").should eq([7.0_f32])
      ensure
        sf.close
      end
    ensure
      FileUtils.rm_rf(dir)
    end
  end

  it "selects a ShardedFile when only an index is present" do
    dir = File.tempname("sharded")
    Dir.mkdir_p(dir)
    begin
      write_safetensors(File.join(dir, "model-00001-of-00001.safetensors"), {"w" => [8.0_f32, 9.0_f32]})
      File.write(File.join(dir, "model.safetensors.index.json"),
        {"weight_map" => {"w" => "model-00001-of-00001.safetensors"}}.to_json)
      sf = SHAInet::HFLoader.open_safetensors(dir)
      begin
        sf.should be_a(SHAInet::SafeTensors::ShardedFile)
        sf.read_f32("w").should eq([8.0_f32, 9.0_f32])
      ensure
        sf.close
      end
    ensure
      FileUtils.rm_rf(dir)
    end
  end

  it "raises when no weights are present" do
    dir = File.tempname("empty")
    Dir.mkdir_p(dir)
    begin
      expect_raises(Exception, /No model.safetensors/) do
        SHAInet::HFLoader.open_safetensors(dir)
      end
    ensure
      FileUtils.rm_rf(dir)
    end
  end
end
