# SHAInet

[![Build Status](https://travis-ci.org/NeuraLegion/shainet.svg?branch=master)](https://travis-ci.org/NeuraLegion/shainet)
[![Join the chat at https://gitter.im/shainet/Lobby](https://badges.gitter.im/shainet/Lobby.svg)](https://gitter.im/shainet/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


SHAInet - stands for Super Human Artificial Intelligence network
a neural network in pure [Crystal](https://crystal-lang.org/)  

At the [Roadmap](https://github.com/NeuraLegion/shainet#development) you can see what we plan to add to the network as the project will progress.  


## Installation

Add this to your application's `shard.yml`:

```yaml
dependencies:
  shainet:
    github: NeuraLegion/shainet
```

## Usage

Standard training on XOR example  
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
xor.add_layer(:input, 2, :memory, SHAInet.sigmoid)
# Add a new layer of the hidden type with 2 neurons and classic neuron type (memory)
xor.add_layer(:hidden, 2, :memory, SHAInet.sigmoid)
# Add a new layer of the output type with 1 neurons and classic neuron type (memory)
xor.add_layer(:output, 1, :memory, SHAInet.sigmoid)
# Fully connect the network layers
xor.fully_connect

# data, training_type, cost_function, activation_function, epochs, error_threshold (sum of errors), learning_rate, momentum)
xor.train(training_data, :sgdm, :mse, 10000, 0.001)

# Run the trained network
xor.run([0, 0])
```


Batch training on the iris dataset using adam
```crystal
# Create a new Data object based on a CSV
  data = SHAInet::Data.new_with_csv_input_target("iris.csv", 0..3, 4)

  # Split the data in a training set and a test set
  training_set, test_set = data.split(0.67)

  # Initiate a new network
  iris = SHAInet::Network.new

  # Add layers
  iris.add_layer(:input, 4, :memory, SHAInet.sigmoid)
  iris.add_layer(:hidden, 5, :memory, SHAInet.sigmoid)
  iris.add_layer(:output, 3, :memory, SHAInet.sigmoid)
  iris.fully_connect

  # Train the network
  iris.train_batch(training_set, :adam, :mse, 30000, 0.000000001)

  # Test the ANN performance
  iris.test(test_set)
```

## Development

### Basic Features  
  - [x] Add sgd,minibatch-update.  
  - [x] Add more activation functions.  
  - [ ] Add more cost functions.  
  - [x] Add more gradient optimizers
    - [x] Add iRprop+  
    - [x] ADAM  

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
- [drujensen](https://github.com/drujensen) - contributor
