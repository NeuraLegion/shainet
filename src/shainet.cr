require "log"

{% if flag?(:enable_cuda) %}
require "./shainet/cuda"
require "./shainet/cudnn"
{% else %}
require "./shainet/cuda_stub"
{% end %}

require "./shainet/autograd/tensor"
require "./shainet/basic/exceptions"
require "./shainet/basic/matrix_layer"
require "./shainet/basic/network_run"
require "./shainet/basic/network_setup"
require "./shainet/concurrency/concurrent_tokenizer"
require "./shainet/data/data"
require "./shainet/data/json_data"
require "./shainet/data/streaming_data"
require "./shainet/data/test_data"
require "./shainet/data/training_data"
require "./shainet/math/batch_processor"
require "./shainet/math/cuda_matrix"
require "./shainet/math/cuda_matrix_ext"
require "./shainet/math/cuda_tensor_matrix"
require "./shainet/math/functions"
require "./shainet/math/gpu_memory"
require "./shainet/math/random_normal"
require "./shainet/math/simple_matrix"
require "./shainet/math/tensor_matrix"
require "./shainet/math/unified_matrix"
require "./shainet/pytorch_import"
require "./shainet/text/bpe_tokenizer"
require "./shainet/text/embedding_layer"
require "./shainet/text/tokenizer"
require "./shainet/transformer/dropout"
require "./shainet/transformer/ext"
require "./shainet/transformer/layer_norm"
require "./shainet/transformer/multi_head_attention"
require "./shainet/transformer/positional_encoding"
require "./shainet/transformer/positionwise_ff"
require "./shainet/transformer/transformer_block"
require "./shainet/version"

module SHAInet
  Log = ::Log.for(self)
  alias GenNum = Float64 | Int32 | Int64 | Float32

  lvl = {
    "info"  => ::Log::Severity::Info,
    "debug" => ::Log::Severity::Debug,
    "warn"  => ::Log::Severity::Warn,
    "error" => ::Log::Severity::Error,
    "fatal" => ::Log::Severity::Fatal,
    "trace" => ::Log::Severity::Trace,
  }

  log_level = (ENV["LOG_LEVEL"]? || "info")

  ::Log.setup(lvl[log_level.downcase])
end
