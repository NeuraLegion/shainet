require "log"
require "json"
require "../pytorch_import"
require "../math/simple_matrix"

module SHAInet
  class Network
    # Notes:
    # ------------
    # There are no matrices in this implementation, instead the gradient values
    # are stored in each neuron/synapse independently.
    # When preforming propogation,
    # all the math is done iteratively on each neuron/synapse locally.
    #
    # This file contains all the methods for creating and maintaining
    # the network, for methods regarding running and training go to network_run.cr
    # ------------

    Log = ::Log.for(self)

    LAYER_TYPES      = ["input", "hidden", "recurrent", "output"]
    CONNECTION_TYPES = ["full", "ind_to_ind", "random"]
    COST_FUNCTIONS   = ["mse", "c_ent", "c_ent_sm"] # , "exp", "hel_d", "kld", "gkld", "ita_sai_d"]

    # General network parameters
    getter :input_layers, :output_layers, :hidden_layers, :recurrent_layers, :lstm_layers, :all_neurons, :all_synapses
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
      @input_layers = Array(Layer).new
      @output_layers = Array(Layer).new
      @hidden_layers = Array(Layer).new
      @recurrent_layers = Array(RecurrentLayer).new
      @lstm_layers = Array(LSTMLayer).new
      @transformer_layers = Array(TransformerLayer).new
      @all_neurons = Array(Neuron).new   # Array of all current neurons in the network
      @all_synapses = Array(Synapse).new # Array of all current synapses in the network
      @error_signal = Array(Float64).new # Array of errors for each neuron in the output layers, based on specific input
      @total_error = 1_f64               # Sum of errors from output layer, based on a specific input
      @mse = 1_f64                       # MSE of netwrok, based on all errors of output layer for a specific input or batch
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

    # Create and populate a layer with neurons
    # l_type is: :input, :hidden or :output
    # l_size = how many neurons in the layer
    # n_type = advanced option for different neuron types
    def add_layer(l_type : Symbol | String, l_size : Int32, n_type : Symbol | String = "memory", activation_function : ActivationFunction = SHAInet.sigmoid, num_heads : Int32 = 1, ff_hidden : Int32 = l_size*4, drop_percent : Int32 = 0, blocks : Int32 = 1, *, vocab_size : Int32 = 0)
      if l_type.to_s == "transformer" && blocks > 1
        blocks.times do
          add_layer(l_type, l_size, n_type, activation_function, num_heads, ff_hidden, drop_percent, 1)
        end
        return
      end
      layer = case l_type.to_s
              when "recurrent"
                RecurrentLayer.new(n_type.to_s, l_size, activation_function)
              when "lstm"
                LSTMLayer.new(n_type.to_s, l_size, activation_function)
              when "embedding"
                raise NeuralNetRunError.new("vocab_size required for embedding layer") if vocab_size <= 0
                EmbeddingLayer.new(vocab_size, l_size, activation_function)
              when "transformer"
                TransformerLayer.new(l_size, num_heads, ff_hidden, drop_percent)
              else
                Layer.new(n_type.to_s, l_size, activation_function)
              end
      unless layer.is_a?(TransformerLayer)
        layer.neurons.each do |neuron|
          @all_neurons << neuron # To easily access neurons later
        end
      end
      if layer.is_a?(RecurrentLayer) || layer.is_a?(LSTMLayer)
        layer.neurons.each do |neuron|
          neuron.synapses_in.each { |s| @all_synapses << s }
        end
      end

      case l_type.to_s
      when "input"
        @input_layers << layer
      when "hidden"
        @hidden_layers << layer
      when "recurrent"
        @hidden_layers << layer
        @recurrent_layers << layer.as(RecurrentLayer)
      when "lstm"
        @hidden_layers << layer
        @lstm_layers << layer.as(LSTMLayer)
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

    # Connect two specific layers with synapses
    def connect_ltl(src_layer : Layer, dest_layer : Layer, connection_type : Symbol | String)
      raise NeuralNetInitalizationError.new("Error initilizing network, must choose correct connection type.") if CONNECTION_TYPES.any? { |x| x == connection_type.to_s } == false
      case connection_type.to_s
      # Connect each neuron from source layer to all neurons in destination layer
      when "full"
        # Resize the weights matrix based on the connecting layer
        if src_layer.is_a?(TransformerLayer)
          dest_layer.weights = Matrix(Float64).build(dest_layer.size, src_layer.size) { rand(-0.1_f64..0.1_f64) }
          dest_layer.biases = Matrix(Float64).build(dest_layer.size, 1) { rand(-0.1_f64..0.1_f64) }
        else
          dest_layer.weights = Matrix(Float64).build(dest_layer.size, src_layer.size) { 0.0 }
        end

        src_layer.neurons.each_with_index do |src_neuron, src_i|
          dest_layer.neurons.each_with_index do |dest_neuron, dest_i|
            synapse = Synapse.new(src_neuron, dest_neuron)
            src_neuron.synapses_out << synapse
            dest_neuron.synapses_in << synapse
            @all_synapses << synapse

            dest_layer.weights[dest_i, src_i] = synapse.weight

            # weights_vector << pointerof(synapse.weight)
            # prev_weights_vector << pointerof(synapse.prev_weight)
            # w_grad_vector << pointerof(synapse.gradient)
          end
        end
        # Connect each neuron from source layer to neuron with
        # corresponding index in destination layer
        # Matrix training is not implemented yet for this connection
      when "ind_to_ind"
        raise NeuralNetInitalizationError.new(
          "Error initializing network, index to index connection requires layers of same size.") if src_layer.neurons.size != dest_layer.neurons.size
        (0..src_layer.neurons.size).each do |index|
          synapse = Synapse.new(src_layer.neurons[index], dest_layer.neurons[index])
          src_layer.neurons[index].synapses_out << synapse
          dest_layer.neurons[index].synapses_in << synapse
          @all_synapses << synapse
        end

        # Randomly decide if each neuron from source layer will
        # connect to a neuron from destination layer
        # Matrix training is not implemented yet for this connection
      when "random"
        src_layer.neurons.each do |src_neuron|     # Source neuron
          dest_layer.neurons.each do |dest_neuron| # Destination neuron
            x = rand(0..1)
            if x <= 0.5 # Currently set to 50% chance, this can be changed at will
              synapse = Synapse.new(src_neuron, dest_neuron)
              src_neuron.synapses_out << synapse
              dest_neuron.synapses_in << synapse
              @all_synapses << synapse
            end
          end
        end
      end
      @all_synapses.uniq!
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
      current_neuron_number = @all_neurons.size
      @hidden_layers.each do |h_l|
        h_l.neurons.each do |neuron|
          kill = false
          if neuron.bias == 0
            neuron.synapses_in.each do |s|
              if s.weight == 0
                kill = true
              end
            end
          end
          if kill
            # Kill neuron and all connected synapses
            neuron.synapses_in.each { |s| @all_synapses.delete(s) }
            neuron.synapses_out.each { |s| @all_synapses.delete(s) }
            @all_neurons.delete(neuron)
            h_l.neurons.delete(neuron)
          end
        end
      end
      Log.info { "Cleaned #{current_neuron_number - @all_neurons.size} dead neurons" }
    end

    def verify_net_before_train
      if @input_layers.empty?
        raise NeuralNetRunError.new("No input layers defined")
        # elsif @hidden_layers.empty?
        #   raise NeuralNetRunError.new("Need atleast one hidden layer")
      elsif @output_layers.empty?
        raise NeuralNetRunError.new("No output layers defined")
      end
      @lstm_layers.each &.setup_gate_params
    end

    def randomize_all_weights
      raise NeuralNetRunError.new("Cannot randomize weights without synapses") if @all_synapses.empty?
      @all_synapses.each &.randomize_weight
    end

    def randomize_all_biases
      raise NeuralNetRunError.new("Cannot randomize biases without neurons") if @all_synapses.empty?
      @all_neurons.each &.randomize_bias
    end

    def save_to_file(file_path : String)
      dump_network = Array(Hash(String, String | Array(Hash(String, Array(Hash(String, String | Float64)) | Float64 | String | String)))).new

      [@input_layers, @output_layers, @hidden_layers].flatten.each do |layer|
        dump_layer = Hash(String, String | Array(Hash(String, Array(Hash(String, String | Float64)) | Float64 | String | String))).new
        dump_neurons = Array(Hash(String, Array(Hash(String, String | Float64)) | Float64 | String | String)).new
        layer.neurons.each do |neuron|
          n = Hash(String, Array(Hash(String, String | Float64)) | Float64 | String | String).new
          n["id"] = neuron.id
          n["bias"] = neuron.bias
          n["n_type"] = neuron.n_type.to_s
          n["synapses_in"] = Array(Hash(String, String | Float64)).new
          n["synapses_out"] = Array(Hash(String, String | Float64)).new
          neuron.synapses_in.each do |s|
            s_h = Hash(String, String | Float64).new
            s_h["source"] = s.source_neuron.id
            s_h["destination"] = s.dest_neuron.id
            s_h["weight"] = s.weight
            n["synapses_in"].as(Array(Hash(String, String | Float64))) << s_h
          end
          neuron.synapses_out.each do |s|
            s_h = Hash(String, String | Float64).new
            s_h["source"] = s.source_neuron.id
            s_h["destination"] = s.dest_neuron.id
            s_h["weight"] = s.weight
            n["synapses_out"].as(Array(Hash(String, String | Float64))) << s_h
          end
          dump_neurons << n
        end

        l_type = ""
        if @input_layers.includes?(layer)
          l_type = "input"
        elsif @hidden_layers.includes?(layer)
          l_type = "hidden"
        else
          l_type = "output"
        end

        dump_layer["l_type"] = l_type
        dump_layer["neurons"] = dump_neurons
        dump_layer["activation_function"] = layer.activation_function.to_s
        dump_network << dump_layer
      end
      File.write(file_path, {"layers" => dump_network}.to_json)
      Log.info { "Network saved to: #{file_path}" }
    end

    def load_from_file(file_path : String)
      net = NetDump.from_json(File.read(file_path))
      net.layers.each do |layer|
        l = Layer.new("memory", 0)
        layer.neurons.each do |neuron|
          n = Neuron.new(neuron.n_type, neuron.id)
          n.bias = neuron.bias
          l.neurons << n
          @all_neurons << n
        end
        case layer.l_type
        when "input"
          @input_layers << l
        when "output"
          @output_layers << l
        when "hidden"
          @hidden_layers << l
        end
      end
      net.layers.each do |layer|
        layer.neurons.each do |n|
          n.synapses_in.each do |s|
            source = @all_neurons.find { |i| i.id == s.source }
            destination = @all_neurons.find { |i| i.id == s.destination }
            next unless source && destination
            _s = Synapse.new(source, destination)
            _s.weight = s.weight
            neuron = @all_neurons.find { |i| i.id == n.id }
            next unless neuron
            neuron.not_nil!.synapses_in << _s
            @all_synapses << _s
          end
          n.synapses_out.each do |s|
            source = @all_neurons.find { |i| i.id == s.source }
            destination = @all_neurons.find { |i| i.id == s.destination }
            next unless source && destination
            _s = Synapse.new(source, destination)
            _s.weight = s.weight
            neuron = @all_neurons.find { |i| i.id == n.id }
            next unless neuron
            neuron.not_nil!.synapses_out << _s
            @all_synapses << _s
          end
        end
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
          mha.w_q = TensorMatrix.from_a(lookup["#{prefix}.mha.w_q"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          mha.w_k = TensorMatrix.from_a(lookup["#{prefix}.mha.w_k"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          mha.w_v = TensorMatrix.from_a(lookup["#{prefix}.mha.w_v"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose
          mha.w_o = TensorMatrix.from_a(lookup["#{prefix}.mha.w_o"]["weight"].as_a.map { |r| r.as_a.map(&.as_f) }).transpose

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
          target.neurons.each_with_index do |neuron, i|
            neuron.bias = bias[i].as_f
            neuron.synapses_in.each_with_index do |syn, j|
              syn.weight = weights[i].as_a[j].as_f
            end
          end
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
          target.neurons.each_with_index do |neuron, i|
            neuron.bias = bias[i].as_f
            neuron.synapses_in.each_with_index do |syn, j|
              syn.weight = weights[i].as_a[j].as_f
            end
          end
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
      Log.info { @all_synapses }
      Log.info { "--------------------------------" }
    end
  end
end
