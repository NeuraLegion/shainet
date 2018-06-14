require "./spec_helper"

# Extract data
system("cd #{__DIR__}/test_data && tar xvf tests.tar.xz")

describe SHAInet::Network do
  puts "############################################################"
  it "can pass an integration test predicting >90% on the iris data set" do
    puts "\n"
    # Create a new Data object based on a CSV
    data = SHAInet::Data.new_with_csv_input_target(__DIR__ + "/test_data/iris.csv", 0..3, 4)

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
    iris.train(
      data: training_set,
      training_type: :adam,
      cost_function: :mse,
      epochs: 30000,
      error_threshold: 1e-9,
      mini_batch_size: 4,
      log_each: 1000,
      show_slice: false)

    # Test the ANN performance
    iris.test(test_set).should be > 0.85
  end
end

# Remove test data
system("cd #{__DIR__}/test_data && rm *.csv")
