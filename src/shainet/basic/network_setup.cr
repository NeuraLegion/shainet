require "log"
require "json"
require "../pytorch_import"
require "../math/simple_matrix"
require "./matrix_layer"

module SHAInet
  class Network
    # Notes:

    #
    # This file contains all the methods for creating and maintaining
    # the network, for methods regarding running and training go to network_run.cr
    # ------------

    Log = ::Log.for(self)

    LAYER_TYPES      = ["input", "hidden", "recurrent", "output"]
    CONNECTION_TYPES = ["full", "ind_to_ind", "random"]
    COST_FUNCTIONS   = ["mse", "c_ent", "c_ent_sm"] # , "exp", "hel_d", "kld", "gkld", "ita_sai_d"]

    # General network parameters
    getter :input_layers, :output_layers, :hidden_layers, :recurrent_layers, :lstm_layers
    getter :transformer_layers
    getter transformer_error : SimpleMatrix
    getter error_signal : Array(Float64), total_error : Float64, :mse, w_gradient : Array(Float64), b_gradient : Array(Float64)

    # Parameters for SGD + Momentum
    property learning_rate : Float64, momentum : Float64

    # Parameters for Rprop
    property etah_plus : Float64, etah_minus : Float64, delta_max : Float64, delta_min : Float64
    getter prev_mse : Float64

    # Parameters for Adam
    property alpha : Float64
    getter beta1 : Float64, beta2 : Float64, epsilon : Float64, time_step : Int32
    property clip_threshold : Float64
    property warmup_steps : Int32
    property weight_decay : Float64
    property accumulation_steps : Int32
    property mixed_precision : Bool

    # First creates an empty shell of the entire network
    def initialize
      @input_layers = Array(MatrixLayer).new
      @output_layers = Array(MatrixLayer).new
      @hidden_layers = Array(MatrixLayer).new
      @transformer_layers = Array(TransformerLayer).new
      @all_layers = Array(MatrixLayer).new
      @error_signal = Array(Float64).new # Array of errors for each element in the output layers
      @total_error = 1_f64               # Sum of errors from output layer, based on a specific input
      @mse = 1_f64                       # MSE of network, based on all errors of output layer for a specific input or batch
      @w_gradient = Array(Float64).new   # Needed for batch train
      @b_gradient = Array(Float64).new   # Needed for batch train

      @learning_rate = 0.005_f64 # Standard parameter for GD
      @momentum = 0.05_f64       # Improved GD

      @etah_plus = 1.2_f64  # For iRprop+ , how to increase step size
      @etah_minus = 0.5_f64 # For iRprop+ , how to decrease step size
      @delta_max = 50_f64   # For iRprop+ , max step size
      @delta_min = 0.1_f64  # For iRprop+ , min step size
      @prev_mse = 1_f64     # For iRprop+ , needed for backtracking

      @alpha = 0.001_f64   # For Adam , step size (recomeneded: only change this hyper parameter when fine-tuning)
      @beta1 = 0.9_f64     # For Adam , exponential decay rate (not recommended to change value)
      @beta2 = 0.999_f64   # For Adam , exponential decay rate (not recommended to change value)
      @epsilon = 10e-8_f64 # For Adam , prevents exploding gradients (not recommended to change value)
      @time_step = 0_i32   # For Adam
      @transformer_error = SimpleMatrix.zeros(1, 1)
      @clip_threshold = Float64::INFINITY
      @warmup_steps = 0
      @weight_decay = 0.0
      @accumulation_steps = 1
      @mixed_precision = false
    end

    # Create and populate a layer
    # l_type is: :input, :hidden or :output
    # l_size = size of the layer
    # n_type = advanced option for layer types
    def add_layer(l_type : Symbol | String, l_size : Int32, n_type : Symbol | String = "memory", activation_function : ActivationFunction = SHAInet.sigmoid, num_heads : Int32 = 1, ff_hidden : Int32 = l_size*4, drop_percent : Int32 = 0, blocks : Int32 = 1, *, vocab_size : Int32 = 0)
      if l_type.to_s == "transformer" && blocks > 1
        blocks.times do
          add_layer(l_type, l_size, n_type, activation_function, num_heads, ff_hidden, drop_percent, 1)
        end
        return
      end
      layer = case l_type.to_s
              when "embedding"
                raise NeuralNetRunError.new("vocab_size required for embedding layer") if vocab_size <= 0
                EmbeddingLayer.new(vocab_size, l_size, activation_function)
              when "transformer"
                TransformerLayer.new(l_size, num_heads, ff_hidden, drop_percent)
              else
                # Use MatrixLayer for regular feedforward layers - it has proper GPU support and gradient computation
                # Note: MatrixLayer will be properly connected with correct input size in connect_ltl
                MatrixLayer.new(1, l_size) # Temporary size, will be updated during connection
              end

      # Add layer to appropriate collections
      case l_type.to_s
      when "input"
        @input_layers << layer
      when "hidden"
        @hidden_layers << layer
      when "embedding"
        @hidden_layers << layer
      when "transformer"
        @hidden_layers << layer
        @transformer_layers << layer.as(TransformerLayer)
      when "output"
        if @output_layers.empty?
          @output_layers << layer
        else
          @output_layers.delete(@output_layers.first)
          @output_layers << layer
          connect_ltl(@hidden_layers.last, @output_layers.first, :full)
        end
      else
        raise NeuralNetRunError.new("Must define correct layer type (:input, :hidden, :recurrent, :lstm, :embedding, :transformer, :output).")
      end

      # Add to all_layers collection
      @all_layers << layer
    end

    # Connect all the layers in order (input and output don't connect between themselves): input, hidden, output
    def fully_connect
      if @hidden_layers.empty?
        # Connect all input layers to all output layers
        @output_layers.each do |out_layer|
          @input_layers.each do |in_layer|
            connect_ltl(in_layer, out_layer, :full)
          end
        end
      else
        # Connect all input layers to the first hidden layer
        @input_layers.each do |in_layer|
          connect_ltl(in_layer, @hidden_layers.first, :full)
        end

        # Connect all hidden layer between each other hierarchically
        (@hidden_layers.size).times do |l|
          next if (l + 1) == @hidden_layers.size
          connect_ltl(@hidden_layers[l], @hidden_layers[l + 1], :full)
        end

        # Connect last hidden layer to all output layers
        @output_layers.each do |out_layer|
          connect_ltl(@hidden_layers.last, out_layer, :full)
        end
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error fully connecting network: #{e}")
    end

    # Connect two specific layers
    def connect_ltl(src_layer : MatrixLayer, dest_layer : MatrixLayer, connection_type : Symbol | String)
      raise NeuralNetInitalizationError.new("Error initilizing network, must choose correct connection type.") if CONNECTION_TYPES.any? { |x| x == connection_type.to_s } == false
      case connection_type.to_s
      # Connect source layer to destination layer with full connections
      when "full"
        # Matrix-based layers handle weight initialization internally
        # Use SimpleMatrix for now to avoid CUDA type mismatches during debugging
        mat_klass = SimpleMatrix
        if src_layer.is_a?(TransformerLayer)
          # For transformer output, weights need to be (d_model, vocab_size) for correct matrix multiplication
          # (batch_size x d_model) * (d_model x vocab_size) = (batch_size x vocab_size)
          dest_layer.weights = mat_klass.new(src_layer.size, dest_layer.size).random_fill!
          dest_layer.biases = mat_klass.new(1, dest_layer.size).random_fill!
        elsif dest_layer.is_a?(MatrixLayer)
          # For MatrixLayer, reinitialize with correct dimensions
          dest_layer.weights = mat_klass.new(src_layer.size, dest_layer.size).random_fill!
          dest_layer.biases = mat_klass.new(1, dest_layer.size).random_fill!
          dest_layer.g_w = mat_klass.zeros(src_layer.size, dest_layer.size)
          dest_layer.g_b = mat_klass.zeros(1, dest_layer.size)
        else
          # Initialize weights randomly for all layer types
          dest_layer.weights = mat_klass.new(dest_layer.size, src_layer.size).random_fill!
          dest_layer.biases = mat_klass.new(dest_layer.size, 1).random_fill!
        end
      end
    rescue e : Exception
      raise NeuralNetRunError.new("Error in connect_ltl: #{e}")
    end

    def log_summary(e)
      Log.info { "Epoch: #{e}, Total error: #{@total_error}, MSE: #{@mse}" }
    end

    def reset_recurrent_state
      @recurrent_layers.each(&.reset_state)
      @lstm_layers.each(&.reset_state)
    end

    def clean_dead_neurons
      # Matrix-based layers don't require cleanup
      # This method is kept for API compatibility
      Log.info { "Matrix-based layers don't require cleanup" }
    end

    def verify_net_before_train
      if @input_layers.empty?
        raise NeuralNetRunError.new("No input layers defined")
        # elsif @hidden_layers.empty?
        #   raise NeuralNetRunError.new("Need atleast one hidden layer")
      elsif @output_layers.empty?
        raise NeuralNetRunError.new("No output layers defined")
      end
    end

    def randomize_all_weights
      # Matrix-based layers handle weight initialization during layer creation
      [@input_layers, @hidden_layers, @output_layers].flatten.each do |layer|
        if layer.weights.is_a?(SimpleMatrix)
          layer.weights.random_fill!
        end
      end
    end

    def randomize_all_biases
      # Matrix-based layers handle bias initialization during layer creation
      [@input_layers, @hidden_layers, @output_layers].flatten.each do |layer|
        if layer.biases.is_a?(SimpleMatrix)
          layer.biases.random_fill!
        end
      end
    end

    def save_to_file(file_path : String)
      dump_network = [] of Hash(String, JSON::Any)

      [@input_layers, @hidden_layers, @output_layers].flatten.each do |layer|
        dump_layer = Hash(String, JSON::Any).new
        l_type = if @input_layers.includes?(layer)
                   "input"
                 elsif @hidden_layers.includes?(layer)
                   "hidden"
                 else
                   "output"
                 end

        dump_layer["l_type"] = JSON::Any.new(l_type)
        if layer.weights
          dump_layer["weights"] = JSON.parse(layer.weights.not_nil!.to_a.to_json)
          dump_layer["biases"] = JSON.parse(layer.biases.not_nil!.to_a.flatten.to_json)
        end
        dump_layer["activation_function"] = JSON::Any.new(layer.activation_function.to_s)
        dump_network << dump_layer
      end

      File.write(file_path, {"layers" => dump_network}.to_json)
      Log.info { "Network saved to: #{file_path}" }
    end

    def load_from_file(file_path : String)
      data = JSON.parse(File.read(file_path))
      layers = data["layers"].as_a

      layers.each do |layer_data|
        l_type = layer_data["l_type"].as_s
        size = layer_data["biases"].as_a.size
        case l_type
        when "input"
          add_layer(:input, size)
        when "output"
          add_layer(:output, size)
        else
          add_layer(:hidden, size)
        end
      end

      fully_connect

      all_layers = @hidden_layers + @output_layers
      layers.each_with_index do |layer_data, idx|
        next if idx == 0 # input layer has no weights to set
        dest_layer = all_layers[idx - 1]
        w = layer_data["weights"].as_a.map { |r| r.as_a.map(&.as_f) }
        b = layer_data["biases"].as_a.map(&.as_f)
        dest_layer.weights = SimpleMatrix.from_a(w)
        dest_layer.biases = SimpleMatrix.from_a([b])
      end
      Log.info { "Network loaded from: #{file_path}" }
    end

    # Load a network from a TorchScript file exported via PyTorch.
    # Supports simple sequential Linear models as well as Transformer
    # models consisting of an embedding layer followed by one or more
    # TransformerLayer blocks and a final Linear output layer.
    def load_from_pt(file_path : String)
      data = PyTorchImport.load(file_path)
      layers = data["layers"].as_a

      lookup = Hash(String, JSON::Any).new
      layers.each { |l| lookup[l["name"].as_s] = l }

      blocks = if blk = data["blocks"]?
                 blk.as_a.map(&.as_s)
               else
                 prefixes = [] of String
                 lookup.keys.each do |k|
                   if m = k.match(/^((?:layers?\.\d+)|(?:layer\d*)|layer)\./)
                     prefix = m[1]
                     prefixes << prefix unless prefixes.includes?(prefix)
                   end
                 end
                 prefixes
               end

      if lookup.has_key?("embedding")
        # Transformer style model. Multiple transformer blocks are
        # supported and will be stacked in the same order as in the
        # exported PyTorch model.
        emb_w = lookup["embedding"]["weight"].as_a
        d_model = emb_w.first.as_a.size
        out_size = lookup["out"]? ? lookup["out"]["weight"].as_a.size : d_model

        add_layer(:input, 1)
        add_layer(:embedding, d_model, vocab_size: emb_w.size)
        blocks.each do
          add_layer(:transformer, d_model)
        end
        add_layer(:output, out_size, activation_function: SHAInet.identity)
        fully_connect

        emb_layer = @hidden_layers.find(&.is_a?(EmbeddingLayer)).as(EmbeddingLayer)
        emb_w.each_with_index do |row, idx|
          row.as_a.each_with_index do |val, j|
            emb_layer.embeddings[idx, j] = val.as_f
          end
        end

        blocks.each_with_index do |prefix, idx|
          t_layer = @transformer_layers[idx]
          mha = t_layer.mha
          mha.w_q = SimpleMatrix.from_a(lookup["#{prefix}.mha.w_q"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          mha.w_k = SimpleMatrix.from_a(lookup["#{prefix}.mha.w_k"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          mha.w_v = SimpleMatrix.from_a(lookup["#{prefix}.mha.w_v"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          mha.w_o = SimpleMatrix.from_a(lookup["#{prefix}.mha.w_o"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose

          ffn = t_layer.ffn
          ffn.w1 = SimpleMatrix.from_a(lookup["#{prefix}.ffn.w1"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          ffn.b1 = SimpleMatrix.from_a([lookup["#{prefix}.ffn.w1"]["bias"].as_a.map(&.as_f)])
          ffn.w2 = SimpleMatrix.from_a(lookup["#{prefix}.ffn.w2"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          ffn.b2 = SimpleMatrix.from_a([lookup["#{prefix}.ffn.w2"]["bias"].as_a.map(&.as_f)])

          n1 = t_layer.norm1
          n1.gamma = SimpleMatrix.from_a([lookup["#{prefix}.norm1"]["weight"].as_a.map(&.as_f)])
          n1.beta = SimpleMatrix.from_a([lookup["#{prefix}.norm1"]["bias"].as_a.map(&.as_f)])
          n2 = t_layer.norm2
          n2.gamma = SimpleMatrix.from_a([lookup["#{prefix}.norm2"]["weight"].as_a.map(&.as_f)])
          n2.beta = SimpleMatrix.from_a([lookup["#{prefix}.norm2"]["bias"].as_a.map(&.as_f)])
        end

        if out = lookup["out"]?
          weights = out["weight"].as_a
          bias = out["bias"].as_a
          target = @output_layers.first
          # Set weights and biases using matrix operations
          w = weights.map { |r| r.as_a.map(&.as_f) }
          b = bias.map(&.as_f)
          target.weights = SimpleMatrix.from_a(w)
          target.biases = SimpleMatrix.from_a([b])
        end
      else
        # Sequential linear model
        input_size = layers.first["weight"].as_a.first.as_a.size
        add_layer(:input, input_size)

        layers.each_with_index do |l, idx|
          out_size = l["weight"].as_a.size
          if idx == layers.size - 1
            add_layer(:output, out_size, activation_function: SHAInet.identity)
          else
            add_layer(:hidden, out_size, activation_function: SHAInet.relu)
          end
        end
        fully_connect

        target_layers = @hidden_layers + @output_layers
        layers.each_with_index do |l, idx|
          weights = l["weight"].as_a
          bias = l["bias"].as_a
          target = target_layers[idx]
          # Set weights and biases using matrix operations
          w = weights.map { |r| r.as_a.map(&.as_f) }
          b = bias.map(&.as_f)
          target.weights = SimpleMatrix.from_a(w)
          target.biases = SimpleMatrix.from_a([b])
        end
      end
    end

    def inspect
      Log.info { @input_layers }
      Log.info { "--------------------------------" }
      Log.info { @hidden_layers }
      Log.info { "--------------------------------" }
      Log.info { @output_layers }
      Log.info { "--------------------------------" }
    end

    # Dummy layers property for compatibility with the matrix-based Network class
    getter layers : Array(MatrixLayer) { [] of MatrixLayer }
  end
end
