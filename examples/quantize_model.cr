require "../src/shainet"
require "json"

# Quantize a model's weights to Q4 once and save as a cache directory.
#
# Usage:
#   crystal run examples/quantize_model.cr -Denable_cuda -- /path/to/model-dir
#
# Creates /path/to/model-dir/.q4/ with the packed Q4 arrays for every 2-D weight. On the next
# load, if HFLoader sees this directory it reads the pre-packed bytes directly -- skipping the
# bf16->fp32->Q4 pipeline that takes ~295 s on a 9B.
#
# Only needs CUDA for the Q4CudaMatrix.pack routine (which is pure CPU maths but lives in a class
# gated on CUDA). Works on any architecture HFLoader supports.

abort "Usage: #{PROGRAM_NAME} <model-dir>" unless ARGV.size >= 1
model_dir = ARGV[0]

cache_dir = File.join(model_dir, ".q4")
manifest_path = File.join(cache_dir, "manifest.json")

if File.exists?(manifest_path)
  STDERR.puts "Already quantized: #{manifest_path}"
  exit 0
end

abort "config.json not found in #{model_dir}" unless File.exists?(File.join(model_dir, "config.json"))
STDERR.puts "Quantizing #{model_dir}..."

Dir.mkdir_p(cache_dir)
t0 = Time.monotonic

sf = SHAInet::HFLoader.open_safetensors(model_dir)
names = sf.tensor_names.reject { |n| n == "__metadata__" }
STDERR.puts "#{names.size} tensors"

manifest = Hash(String, Hash(String, Int64)).new
saved = 0

names.each_with_index do |name, idx|
  # Try reading as a matrix. 1-D tensors (norms, biases) are not quantized.
  begin
    w = sf.read_matrix_transposed(name)
  rescue
    STDERR.puts "  [#{idx + 1}/#{names.size}] #{name} -- skip (1-D or unsupported)"
    next
  end

  STDERR.print "  [#{idx + 1}/#{names.size}] #{name} [#{w.rows}, #{w.cols}]..."
  STDERR.flush
  q, d, sub = SHAInet::Q4CudaMatrix.pack(w)

  base = name.gsub(".", "__")
  File.open(File.join(cache_dir, "#{base}.q"), "w") { |f| f.write(q.to_unsafe.to_slice(q.size)) }
  File.open(File.join(cache_dir, "#{base}.d"), "w") { |f| f.write(d.to_unsafe.as(Pointer(UInt8)).to_slice(d.size * 4)) }
  File.open(File.join(cache_dir, "#{base}.sub"), "w") { |f| f.write(sub.to_unsafe.to_slice(sub.size)) }

  manifest[name] = {
    "rows"     => w.rows.to_i64,
    "cols"     => w.cols.to_i64,
    "q_size"   => q.size.to_i64,
    "d_size"   => d.size.to_i64,
    "sub_size" => sub.size.to_i64,
  }
  saved += 1
  STDERR.puts " OK"
  GC.collect
end

sf.close

File.write(manifest_path, manifest.to_json)
dt = (Time.monotonic - t0).total_seconds
total_bytes = Dir.children(cache_dir).sum { |f| File.size(File.join(cache_dir, f)) }
STDERR.puts "Done: #{saved} tensors in #{dt.round(1)}s, #{(total_bytes / 1_073_741_824.0).round(2)} GiB"
