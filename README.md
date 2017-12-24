# SHAInet

SHAInet - stands for Super Human Artificial Intelligence network
a neural network in pure Crystal

The current network is a vanilla NN, which supports backprop and feed-forward.
It solves XOR and iris dataset which is part of the network specs.  

## Installation

Add this to your application's `shard.yml`:

```yaml
dependencies:
  shainet:
    github: NeuraLegion/shainet
```

## Usage

```crystal
require "shainet"

training_data = [
  [[0, 0], [0]],
  [[1, 0], [1]],
  [[0, 1], [1]],
  [[1, 1], [0]],
]
# Initialize a new network
xor = SHAInet::Network.new
# Add a new layer of the input type with 2 neurons and classic neuron type (memory)
xor.add_layer(:input, 2, :memory)
# Add a new layer of the hidden type with 2 neurons and classic neuron type (memory)
xor.add_layer(:hidden, 2, :memory)
# Add a new layer of the output type with 1 neurons and classic neuron type (memory)
xor.add_layer(:output, 1, :memory)
# Fully connect the network layers
xor.fully_connect

# data, cost_function, activation_function, epochs, error_threshold
xor.train(training_data, :mse, :sigmoid, 10000, 0.01)

# Run the trained network
xor.run([0, 0])
```


## Development

### Basic Features  
  - [ ] Add sgd,minibatch-update.  
  - [ ] Add rprop/irprop+  
  - [ ] Add more activation functions.  
  - [ ] Add more cost functions.  

### Advanced Features  
  - [ ] Bind and use CUDA (GPU acceleration)  
  - [ ] graphic printout of network architecture.  
  - [ ] Add LSTM.  
    - [ ] RNN.  
  - [ ] Convolutional Neural Net.  
  - [ ] GNG (growing neural gas).  
  - [ ] SOM (self organizing maps).  
  - [ ] DBM (deep belief network).  
  - [ ] Add support for multiple neuron types.  





## Contributing

1. Fork it ( https://github.com/NeuraLegion/shainet/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

## Contributors

- [ArtLinkov](https://github.com/ArtLinkov) - creator, maintainer
- [bararchy](https://github.com/bararchy) - creator, maintainer

