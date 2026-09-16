require "./spec_helper"

# read_matrix_transposed streams a tensor straight into its transposed destination instead of
# reading then transposing. It exists to halve the load-time peak, and it was shipped without
# this comparison -- which is exactly the kind of change that produces plausible-looking numbers
# in the wrong order rather than an error.
#
# Written as real safetensors files so the dtype conversion paths are covered as they are used,
# not through a stub.

def st_header(name : String, dtype : String, shape : Array(Int32), nbytes : Int32) : String
  %({"#{name}":{"dtype":"#{dtype}","shape":[#{shape.join(",")}],"data_offsets":[0,#{nbytes}]}})
end

def st_write(path : String, name : String, dtype : String, shape : Array(Int32), body : Bytes)
  hdr = st_header(name, dtype, shape, body.size)
  pad = (8 - (hdr.bytesize % 8)) % 8
  hdr += " " * pad
  ::File.open(path, "w") do |f|
    len = Bytes.new(8)
    IO::ByteFormat::LittleEndian.encode(hdr.bytesize.to_u64, len)
    f.write(len)
    f.print(hdr)
    f.write(body)
  end
end

def st_f32_body(vals : Array(Float32)) : Bytes
  b = Bytes.new(vals.size * 4)
  vals.each_with_index { |v, i| IO::ByteFormat::LittleEndian.encode(v, b[i * 4, 4]) }
  b
end

# bf16 is the top 16 bits of the f32 bit pattern, which is why the conversion is a shift.
def st_bf16_body(vals : Array(Float32)) : Bytes
  b = Bytes.new(vals.size * 2)
  vals.each_with_index do |v, i|
    bits = v.unsafe_as(UInt32)
    IO::ByteFormat::LittleEndian.encode((bits >> 16).to_u16, b[i * 2, 2])
  end
  b
end

describe "SafeTensors#read_matrix_transposed" do
  it "matches read_matrix(...).transpose element for element on f32" do
    rows, cols = 7, 5
    vals = Array(Float32).new(rows * cols) { |i| (i * 0.37 - 3.0).to_f32 }
    path = ::File.tempname("st", ".safetensors")
    begin
      st_write(path, "w", "F32", [rows, cols], st_f32_body(vals))
      sf = SHAInet::SafeTensors::File.new(path)
      begin
        want = sf.read_matrix("w").transpose
        got = sf.read_matrix_transposed("w")
        got.rows.should eq(cols)
        got.cols.should eq(rows)
        # Exact, not approximate: this is a reordering, so any difference is a wiring bug and a
        # tolerance would hide the one failure mode the example exists to catch.
        cols.times { |i| rows.times { |j| got[i, j].should eq(want[i, j]) } }
      ensure
        sf.close
      end
    ensure
      ::File.delete?(path)
    end
  end

  it "matches read_matrix(...).transpose element for element on bf16" do
    # The real checkpoint is bf16, and this is the branch the 9B load actually takes.
    rows, cols = 6, 9
    vals = Array(Float32).new(rows * cols) { |i| ((i % 13) * 0.25 - 1.5).to_f32 }
    path = ::File.tempname("st", ".safetensors")
    begin
      st_write(path, "w", "BF16", [rows, cols], st_bf16_body(vals))
      sf = SHAInet::SafeTensors::File.new(path)
      begin
        want = sf.read_matrix("w").transpose
        got = sf.read_matrix_transposed("w")
        got.rows.should eq(cols)
        got.cols.should eq(rows)
        cols.times { |i| rows.times { |j| got[i, j].should eq(want[i, j]) } }
      ensure
        sf.close
      end
    ensure
      ::File.delete?(path)
    end
  end

  it "distinguishes a transpose from a reshape on a non-square tensor" do
    # A reshape preserves every value and the total count, so a spec that only compared sorted
    # contents or shapes would pass on a reshape bug. This pins one specific off-diagonal cell.
    path = ::File.tempname("st", ".safetensors")
    begin
      # [2, 3] laid out row-major as 1..6, so element (0, 2) is 3 and the transpose puts it at
      # (2, 0). A reshape to [3, 2] would put 3 at (1, 0) instead.
      st_write(path, "w", "F32", [2, 3], st_f32_body([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(&.to_f32)))
      sf = SHAInet::SafeTensors::File.new(path)
      begin
        got = sf.read_matrix_transposed("w")
        got.rows.should eq(3)
        got.cols.should eq(2)
        got[2, 0].should eq(3.0_f32)
        got[0, 1].should eq(4.0_f32)
        got[1, 0].should eq(2.0_f32)
      ensure
        sf.close
      end
    ensure
      ::File.delete?(path)
    end
  end

  it "returns a 1-D tensor unchanged, since a vector has no orientation to swap" do
    path = ::File.tempname("st", ".safetensors")
    begin
      st_write(path, "n", "F32", [4], st_f32_body([1.5, 2.5, 3.5, 4.5].map(&.to_f32)))
      sf = SHAInet::SafeTensors::File.new(path)
      begin
        got = sf.read_matrix_transposed("n")
        want = sf.read_matrix("n")
        got.rows.should eq(want.rows)
        got.cols.should eq(want.cols)
        want.cols.times { |j| got[0, j].should eq(want[0, j]) }
      ensure
        sf.close
      end
    ensure
      ::File.delete?(path)
    end
  end
end
