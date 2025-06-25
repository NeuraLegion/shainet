require "../src/shainet"

# Example of loading HuggingFace GPT weights.
# Provide the path to `pytorch_model.bin` as the first argument.

unless ARGV.size > 0
  puts "usage: crystal examples/hf_gpt_import.cr <pytorch_model.bin>"
  exit
end

net = SHAInet::Network.new
net.load_from_pt(ARGV[0])

puts "Loaded #{net.transformer_layers.size} transformer blocks"
