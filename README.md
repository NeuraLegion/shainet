# SHAInet

SHAInet - stands for Super Human Artificial Intelligence network
a neural network in pure [Crystal](https://crystal-lang.org/)

This is a free-time project, happily hosted by BrightSec that was created as part of some internal research. We started it with research in mind, rather than production, and just kept going, also thanks to members of the community.

The original version of SHAInet was created with the goal of testing biologically inspired neural networks, but it has since evolved into a more general-purpose neural network library.

The latest versions of SHAInet are designed to be used for training and running neural networks, with a focus on simplicity and ease of use. It supports various types of layers, activation functions, and training algorithms.

## Installation

Add this to your application's `shard.yml`:

```yaml
dependencies:
  shainet:
    github: NeuraLegion/shainet
```

### Optional CUDA setup

To enable GPU acceleration install the CUDA Toolkit so that `libcudart.so` and
`libcublas.so` are reachable in your `LD_LIBRARY_PATH`. SHAInet will
automatically detect these libraries at runtime and switch to GPU matrices when
available. When CUDA cannot be loaded, training falls back to the CPU
implementation.

Verify CUDA support with:

```crystal
require "shainet"
puts "CUDA available: #{SHAInet::CUDA.available?}"
puts "CUDA version: #{SHAInet::CUDA.version || "unknown"}"
```

If the libraries are installed in a non-standard location set
`LD_LIBRARY_PATH` accordingly before running the specs or your program:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

You can verify the path with `ldconfig -p | grep libcudart` and add the export line to your shell profile (e.g. `~/.bashrc`) to persist the setting.

No additional build flags are required as the CUDA and cuBLAS libraries are
dynamically loaded at runtime.

cuDNN support is detected in the same way. Ensure `libcudnn.so` can be found in
`LD_LIBRARY_PATH` if you want to use optimized convolution or activation
kernels:

```crystal
puts "cuDNN available: #{SHAInet::CUDA.cudnn_available?}"
```

Optional custom kernels can be provided in `libshainet_cuda_kernels.so`. Their
presence is reported by `SHAInet::CUDA.kernels_available?`.

### Optimized GPU Setup (Recommended)

For maximum GPU performance, especially with transformer models, use the optimized installation that builds custom CUDA kernels:

```bash
# Clone the repository
git clone https://github.com/NeuraLegion/shainet.git
cd shainet

# Install dependencies and build CUDA kernels
make install

# Set library path for GPU acceleration
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)

# Verify GPU acceleration is working
make test
```

This builds custom CUDA kernels that provide significant speedups for:

- Softmax operations (attention mechanisms)
- Layer normalization
- Dropout layers
- Embedding lookups
- Matrix operations

**Performance Impact:** This can improve GPU utilization from ~2% to 60-90% in transformer training, resulting in 3-10x faster training speeds.

**Alternative installation:**

```bash
# Manual kernel building
./build_cuda_kernels.sh
```

Add the library path to your `~/.bashrc` for permanent use:

```bash
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)" >> ~/.bashrc
```

## Usage

More examples live in the `examples/` folder and the specs.  The network API is
built on top of `MatrixLayer` which works on both CPU and GPU.

### XOR network (MatrixLayer)

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

### Iris classification

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

### Training with StreamingData

`StreamingData` reads batches lazily from disk using a small buffer so even
massive corpora can be processed. Each line in the data file should contain a
JSON array describing the input and expected output: `[[1,0],[1]]`. Use the
`chunk_size` argument to control how many lines are buffered and shuffled at a
time.

```crystal
# Buffer at most 1,024 lines and shuffle each chunk
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

When `gpu_batches` is set to `true` and CUDA is available, `next_batch` will
return `CudaMatrix` pairs so the training loop can operate directly on GPU
batches.

### Autograd::Tensor and TensorMatrix

`Autograd::Tensor` wraps a numeric value and tracks how it was computed so
gradients can be propagated automatically. `TensorMatrix` is a lightweight
matrix made of tensors for differentiable operations. Each tensor stores its
value in `data` and accumulates gradients in `grad` during backpropagation.

```crystal
a = SHAInet::SimpleMatrix.tensor(1, 2)
a[0, 0] = SHAInet::Autograd::Tensor.new(2.0)
a[0, 1] = SHAInet::Autograd::Tensor.new(3.0)

w = SHAInet::SimpleMatrix.tensor(2, 1)
w[0, 0] = SHAInet::Autograd::Tensor.new(4.0)
w[1, 0] = SHAInet::Autograd::Tensor.new(5.0)

out = a * w
out[0, 0].as(SHAInet::Autograd::Tensor).backward

learning_rate = 0.1
w.rows.times do |i|
  w.cols.times do |j|
    t = w[i, j]
    w[i, j] = SHAInet::Autograd::Tensor.new(t.data - learning_rate * t.grad)
    t.grad = 0.0
  end
end
```

After calling `backward` the gradients reside in each tensor's `grad` field.
The loop above applies a simple gradient descent step. Use
`TensorMatrix#zero_grads!` to clear gradients when starting a new iteration.

## Development

### Basic Features

- [x] Train network
- [x] Save/load
- [x] Activation functions:
  - [x] Sigmoid
  - [x] Bipolar sigmoid
  - [x] log-sigmoid
  - [x] Tanh
  - [x] ReLU
  - [x] Leaky ReLU
  - [x] Softmax
- [x] Cost functions:
  - [x] Quadratic
  - [x] Cross-entropy
- [x] Gradient optimizers
  - [x] SGD + momentum
  - [x] iRprop+
  - [x] ADAM
  - [x] ES (evolutionary strategy, non-backprop)
- [x] Autosave during training

### Advanced Features

- [x] Support activation functions as Proc
- [x] Support cost functions as Proc
- [x] Embedding layers
- [x] Layer normalization for transformer layers
- [x] Add support for multiple neuron types.
- [x] Bind and use CUDA (GPU acceleration)

### BabyLM Transformer example

The file `examples/babylm_transformer.cr` trains a small Transformer
language model on the BabyLM corpus. After tokenizing the text,
generate streaming pairs with:

```bash
python3 scripts/write_token_pairs.py tokens.txt 16 train_pairs.jsonl
```

Then train using `StreamingData`:

```bash
crystal run examples/babylm_transformer.cr
```

### Loading a PyTorch model

SHAInet can import simple sequential models or a tiny Transformer
exported from PyTorch as TorchScript. First export your model from
Python:

```python
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(2, 3),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 1)
)
example = torch.randn(1, 2)
traced = torch.jit.trace(model, example)
traced.save("model.pt")
```

Then load the file in Crystal:

```crystal
net = SHAInet::Network.new
net.load_from_pt("model.pt")
output = net.run([1.0, 2.0])
```

To create a tiny Transformer model for import you can use the helper
script:

```bash
python3 scripts/build_transformer_model.py transformer.pt
```

Then load it the same way (input is a token id):

```crystal
net = SHAInet::Network.new
net.load_from_pt("transformer.pt")
out = net.run([1])
```

### Loading a HuggingFace GPT model

Weights from models like GPT-2 published on HuggingFace can be loaded
directly from the `pytorch_model.bin` file. The conversion happens
automatically using `scripts/pt_to_json.py`.

```crystal
net = SHAInet::Network.new
net.load_from_pt("pytorch_model.bin")
```

## Contributing

1. Fork it [https://github.com/NeuraLegion/shainet/fork](fork)
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

## Contributors

- [ArtLinkov](https://github.com/ArtLinkov) - creator, maintainer
- [bararchy](https://github.com/bararchy) - creator, maintainer
- [drujensen](https://github.com/drujensen) - contributor
- [hugoabonizio](https://github.com/hugoabonizio) - contributor
- [Rémy Marronnier](https://github.com/rmarronnier) - contributor
- [psikoz](https://github.com/psikoz) - logo desgin
