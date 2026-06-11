## SHAInet - A neural network in pure [Crystal](https://crystal-lang.org/)

SHAInet (Super Human Artificial Intelligence Network) is a neural network library written in pure [Crystal](https://crystal-lang.org/). Originally created for biologically inspired neural network research, it has evolved into a general-purpose library for training and running neural networks, with a focus on simplicity and ease of use.

---

## Features

- CPU and GPU (CUDA) support
- Multiple layer types and activation functions
- Various training algorithms (SGD, Adam, iRprop+, etc.)
- Streaming data support for large datasets
- HuggingFace model import via SafeTensors (no Python required)
- Transformer and modern NLP support

---

## Installation

Add to your `shard.yml`:

```yaml
dependencies:
  shainet:
    github: NeuraLegion/shainet
```

### GPU Acceleration (Optional)

- Install the CUDA Toolkit and ensure `libcudart.so` and `libcublas.so` are in your `LD_LIBRARY_PATH`.
- SHAInet will auto-detect CUDA and use GPU acceleration if available.
- For cuDNN support, ensure `libcudnn.so` is also in your `LD_LIBRARY_PATH`.
- Compile the project with `-Denable_cuda`

Check CUDA availability:

```crystal
require "shainet"
puts "CUDA available: #{SHAInet::CUDA.available?}"
puts "CUDA version: #{SHAInet::CUDA.version || "unknown"}"
```

#### Optimized GPU Setup

For best performance (especially with transformers):

```bash
git clone https://github.com/NeuraLegion/shainet.git
cd shainet
make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
make test
```

To build kernels manually:

```bash
./build_cuda_kernels.sh
```

#### RTX 30/40 Series (Ampere/Ada) Note

These GPUs use TF32 tensor cores by default for FP32 matrix multiply, which
reduces mantissa precision from 23 to 10 bits. For LLM inference this can cause
non-deterministic token generation. SHAInet sets
`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` automatically, but for full
FP32 precision also set:

```bash
export NVIDIA_TF32_OVERRIDE=0
```

### Device management

Layers such as `LayerNorm` allocate workspace matrices on the first forward pass
and reuse them across iterations. Call `to_gpu!` or `to_cpu!` only when
switching devices. Repeated calls without a device change keep the existing
workspaces to avoid unnecessary allocations.

---

## Usage

See `examples/` for more.

### XOR Example

```crystal
require "shainet"

data = [
  [[0, 0], [0]],
  [[1, 0], [1]],
  [[0, 1], [1]],
  [[1, 1], [0]],
]

net = SHAInet::Network.new
net.add_layer(:input, 2)
net.add_layer(:hidden, 2)
net.add_layer(:output, 1)
net.fully_connect

net.train(data: data,
  training_type: :sgdm,
  cost_function: :mse,
  epochs: 5000,
  log_each: 1000)

puts net.run([0, 1])
```

### Load a HuggingFace Model (SafeTensors)

Load any GPT-2 compatible model directly from HuggingFace SafeTensors format.
No Python, no PyTorch — pure Crystal binary parsing.

```crystal
require "shainet"

# Download model.safetensors and config.json from HuggingFace
net = SHAInet::HFLoader.load_gpt2("/path/to/model-dir")

# Run inference — input is token IDs as column vector
input = SHAInet::SimpleMatrix.new(3, 1) # 3 tokens
input[0, 0] = 15.0  # token id
input[1, 0] = 42.0
input[2, 0] = 7.0

output = net.run(input) # => [3, vocab_size] logits
```

Supported formats: F16, BF16, F32, F64. Works with any GPT-2 architecture
model on HuggingFace that provides `model.safetensors`.

### Iris Classification

```crystal
data = SHAInet::Data.new_with_csv_input_target("iris.csv", 0..3, 4)
train, test = data.split(0.67)

iris = SHAInet::Network.new
iris.add_layer(:input, 4)
iris.add_layer(:hidden, 5)
iris.add_layer(:output, 3)
iris.fully_connect

iris.train_batch(
  data: train,
  training_type: :adam,
  cost_function: :mse,
  epochs: 2000,
  log_each: 100)

puts iris.test(test)
```

### Streaming Data

Efficiently train on large datasets:

```crystal
stream = SHAInet::StreamingData.new(
  "data.txt",
  shuffle: true,
  chunk_size: 1024,
  gpu_batches: true)

net = SHAInet::Network.new
net.add_layer(:input, 2, :memory, SHAInet.sigmoid)
net.add_layer(:hidden, 3, :memory, SHAInet.sigmoid)
net.add_layer(:output, 1, :memory, SHAInet.sigmoid)
net.fully_connect

net.train(
  data: stream,
  training_type: :sgdm,
  epochs: 5000,
  mini_batch_size: 2,
  log_each: 1000)
```

---

## Advanced

- Run a real LLaMA model: `crystal run examples/llama_chat.cr -Denable_cuda`
  (auto-downloads Llama-3.2-1B-Instruct, chats with KV cache + GPU).
- Quantized inference (Q8_0): call `net.quantize!` after loading to run with
  int8 weights + per-32-block fp32 scales (dequant-in-kernel GEMV). Cuts weight
  VRAM ~4x (1B model: ~5GB fp32 → ~1.3GB) and speeds up memory-bound decode.
  `llama_chat.cr` quantizes by default on GPU; set `SHAINET_FP32=1` to keep fp32.
  Benchmark/eval both paths with `examples/q8_eval.cr`.
- See `examples/babylm_transformer.cr` for training a transformer language model.
- Use `SHAInet::SafeTensors::File` to read any `.safetensors` file directly.

### SafeTensors API

```crystal
# Low-level tensor access
sf = SHAInet::SafeTensors::File.new("model.safetensors")
sf.tensor_names          # => ["transformer.wte.weight", ...]
info = sf.tensors["transformer.wte.weight"]
info.dtype               # => F32
info.shape               # => [50257, 768]

matrix = sf.read_matrix("transformer.wte.weight")  # => SimpleMatrix
data = sf.read_f64("transformer.h.0.ln_1.weight")  # => Array(Float64)
sf.close
```

### Autograd

```crystal
a = SHAInet::SimpleMatrix.tensor(1, 2)
a[0, 0] = SHAInet::Autograd::Tensor.new(2.0)
a[0, 1] = SHAInet::Autograd::Tensor.new(3.0)

w = SHAInet::SimpleMatrix.tensor(2, 1)
w[0, 0] = SHAInet::Autograd::Tensor.new(4.0)
w[1, 0] = SHAInet::Autograd::Tensor.new(5.0)

out = a * w
out[0, 0].as(SHAInet::Autograd::Tensor).backward
```

## Contributing

1. Fork [https://github.com/NeuraLegion/shainet](https://github.com/NeuraLegion/shainet)
2. Create a feature branch
3. Commit and push your changes
4. Open a Pull Request

---

## Contributors

- [ArtLinkov](https://github.com/ArtLinkov) - creator, maintainer
- [bararchy](https://github.com/bararchy) - creator, maintainer
- [drujensen](https://github.com/drujensen) - contributor
- [hugoabonizio](https://github.com/hugoabonizio) - contributor
- [Rémy Marronnier](https://github.com/rmarronnier) - contributor
- [psikoz](https://github.com/psikoz) - logo design

---
