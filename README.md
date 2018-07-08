<p align="center"><img src="logo/logotype_vertical.png" alt="shainet" height="200px"></p>


[![Build Status](https://travis-ci.org/NeuraLegion/shainet.svg?branch=master)](https://travis-ci.org/NeuraLegion/shainet)
[![Join the chat at https://gitter.im/shainet/Lobby](https://badges.gitter.im/shainet/Lobby.svg)](https://gitter.im/shainet/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


SHAInet - stands for Super Human Artificial Intelligence network
a neural network in pure [Crystal](https://crystal-lang.org/)

This is a free-time project, happily hosted by NeuraLegion that was created as part of some internal research. We started it with research in mind, rather than production, and just kept going, also thanks to members of the community.

We wanted to try and implement some inspiration from the biological world into this project. In addition to that, we wanted to try an approach for NNs using object-oriented modeling instead of matrices. The main reason behind that was, to try new types of neurons aiming for more robust learning (if possible) or at least have more fine-tuned control over the manipulation of each neuron (which is difficult using a matrix-driven approach).

At the [Roadmap](https://github.com/NeuraLegion/shainet#development) you can see what we plan to add to the network as the project will progress.  


## Installation

Add this to your application's `shard.yml`:

```yaml
dependencies:
  shainet:
    github: NeuraLegion/shainet
```

## Usage

More usage examples can be found in the specs

### Standard training on XOR example  
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

# Adjust network parameters
xor.learning_rate = 0.7
xor.momentum = 0.3

# data, training_type, cost_function, activation_function, epochs, error_threshold (sum of errors), learning_rate, momentum)
xor.train(
      data: training_data,
      training_type: :sgdm,
      cost_function: :mse,
      epochs: 5000,
      error_threshold: 0.000001,
      log_each: 1000)

# Run the trained network
xor.run([0, 0])
```


### Batch training on the iris dataset using adam
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

# Adjust network parameters
xor.learning_rate = 0.7
xor.momentum = 0.3

# Train the network
iris.train_batch(
      data: normalized.data.shuffle,
      training_type: :adam,
      cost_function: :mse,
      epochs: 20000,
      error_threshold: 0.000001,
      log_each: 1000)

# Test the network's performance
iris.test(test_set)
```

### Using convolutional network
```crystal

# Load training data (partial dataset)
raw_data = Array(Array(Float64)).new
csv = CSV.new(File.read(__DIR__ + "/test_data/mnist_train.csv"))
10000.times do
  # CSV.each_row(File.read(__DIR__ + "/test_data/mnist_train.csv")) do |row|
  csv.next
  new_row = Array(Float64).new
  csv.row.to_a.each { |value| new_row << value.to_f64 }
  raw_data << new_row
end
raw_input_data = Array(Array(Float64)).new
raw_output_data = Array(Array(Float64)).new

raw_data.each do |row|
  raw_input_data << row[1..-1]
  raw_output_data << [row[0]]
end

training_data = SHAInet::CNNData.new(raw_input_data, raw_output_data)
training_data.for_mnist_conv
training_data.data_pairs.shuffle!

# Load test data (partial dataset)
raw_data = Array(Array(Float64)).new
csv = CSV.new(File.read(__DIR__ + "/test_data/mnist_test.csv"))
1000.times do
  csv.next
  new_row = Array(Float64).new
  csv.row.to_a.each { |value| new_row << value.to_f64 }
  raw_data << new_row
end

raw_input_data = Array(Array(Float64)).new
raw_output_data = Array(Array(Float64)).new

raw_data.each do |row|
  raw_input_data << row[1..-1]
  raw_output_data << [row[0]]
end

# Load data to a CNNData helper class
test_data = SHAInet::CNNData.new(raw_input_data, raw_output_data)
test_data.for_mnist_conv # Normalize and make labels into 'one-hot' vectors

# Initialize Covnolutional network
cnn = SHAInet::CNN.new

# Add layers to the model
cnn.add_input([height = 28, width = 28, channels = 1]) # Output shape = 28x28x1
cnn.add_conv(
  filters_num: 20,
  window_size: 5,
  stride: 1,
  padding: 2,
  activation_function: SHAInet.none)  # Output shape = 28x28x20
cnn.add_relu(0.01)                    # Output shape = 28x28x20
cnn.add_maxpool(pool: = 2, stride = 2) # Output shape = 14x14x20
cnn.add_conv(
  filters_num: 20,
  window_size: 5,
  stride: 1,
  padding: 2,
  activation_function: SHAInet.none)  # Output shape = 14x14x40
cnn.add_maxpool(pool:2, stride: 2)    # Output shape = 7x7x40
cnn.add_fconnect(l_size: 10, activation_function: SHAInet.sigmoid)
cnn.add_fconnect(l_size: 10, activation_function: SHAInet.sigmoid)
cnn.add_softmax

cnn.learning_rate = 0.005
cnn.momentum = 0.02

# Train the model on the training-set
cnn.train_batch(
  data: training_data.data_pairs,
  training_type: :sgdm,
  cost_function: :mse,
  epochs: 3,
  error_threshold: 0.0001,
  log_each: 1,
  mini_batch_size: 50)

# Evaluate accuracy on the test-set
correct_answers = 0
test_data.data_pairs.each do |data_point|
  result = cnn.run(data_point[:input], stealth: true)
  if (result.index(result.max) == data_point[:output].index(data_point[:output].max))
    correct_answers += 1
  end
end

# Print the layer activations
cnn.inspect("activations")
puts "We managed #{correct_answers} out of #{test_data.data_pairs.size} total"
puts "Cnn output: #{cnn.output}"
```

### Evolutionary optimizer example:
```crystal
label = {
      "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
      "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
      "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
    }

    iris = SHAInet::Network.new
    iris.add_layer(:input, 4, :memory, SHAInet.sigmoid)
    iris.add_layer(:hidden, 4, :memory, SHAInet.sigmoid)
    iris.add_layer(:output, 3, :memory, SHAInet.sigmoid)
    iris.fully_connect

    # Get data from a local file
    outputs = Array(Array(Float64)).new
    inputs = Array(Array(Float64)).new
    CSV.each_row(File.read(__DIR__ + "/test_data/iris.csv")) do |row|
      row_arr = Array(Float64).new
      row[0..-2].each do |num|
        row_arr << num.to_f64
      end
      inputs << row_arr
      outputs << label[row[-1]]
    end
    data = SHAInet::TrainingData.new(inputs, outputs)
    data.normalize_min_max

    training_data, test_data = data.split(0.9)

    iris.train_es(
      data: training_data,
      pool_size: 50,
      learning_rate: 0.5,
      sigma: 0.1,
      cost_function: :c_ent,
      epochs: 500,
      mini_batch_size: 15,
      error_threshold: 0.00000001,
      log_each: 100,
      show_slice: true)

    # Test the trained model
    correct = 0
    test_data.data.each do |data_point|
      result = iris.run(data_point[0], stealth: true)
      expected = data_point[1]
      # puts "result: \t#{result.map { |x| x.round(5) }}"
      # puts "expected: \t#{expected}"
      error_sum = 0.0
      result.size.times do |i|
        error_sum += (result[i] - expected[i]).abs
      end
      correct += 1 if error_sum < 0.3
    end
    puts "Correct answers: (#{correct} / #{test_data.size})"
    (correct > 10).should eq(true)
```


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
  - [x] Convolutional Neural Net.  
  - [ ] Add support for multiple neuron types.  
  - [ ] Bind and use CUDA (GPU acceleration)  
  - [ ] graphic printout of network architecture.  
  
### Possible Future Features
  - [ ] RNN (recurant neural network)
  - [ ] LSTM (long-short term memory)
  - [ ] GNG (growing neural gas).  
  - [ ] SOM (self organizing maps).  
  - [ ] DBM (deep belief network).  





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
- [hugoabonizio](https://github.com/hugoabonizio) - contributor
