## SHAInet - A neural network in pure [Crystal](https://crystal-lang.org/)

SHAInet (Super Human Artificial Intelligence Network) is a neural network library written in pure [Crystal](https://crystal-lang.org/). Originally created for biologically inspired neural network research, it has evolved into a general-purpose library for training and running neural networks, with a focus on simplicity and ease of use.

---

## Features

- CPU and GPU (CUDA) support
- Multiple layer types and activation functions
- Various training algorithms (SGD, Adam, iRprop+, etc.)
- Streaming data support for large datasets
- PyTorch and HuggingFace model import
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

---

## Advanced

- See `examples/babylm_transformer.cr` for a transformer language model.
- Import PyTorch models with `net.load_from_pt("model.pt")`.
- Import HuggingFace GPT weights directly from `pytorch_model.bin`.

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
