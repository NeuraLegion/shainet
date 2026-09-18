module SHAInet
  module CUDA
    extend self
    Log = ::Log.for(self)

    enum MemcpyKind
      HostToHost     = 0
      HostToDevice   = 1
      DeviceToHost   = 2
      DeviceToDevice = 3
    end

    enum Operation
      N = 0
      T = 1
    end

    lib LibCUBLAS
      type Handle = Void*
    end

    def available? : Bool
      false
    end

    def fully_available? : Bool
      false
    end

    def version
      nil
    end

    def cudnn_available? : Bool
      false
    end

    def kernels_available? : Bool
      false
    end

    def malloc(*args) : Int32
      raise "CUDA disabled"
    end

    def free(*args)
    end

    def memcpy(*args)
    end

    def copy_device_to_device(*args)
    end

    def device_synchronize
    end

    def malloc_host(*args)
      raise "CUDA disabled"
    end

    def free_host(*args)
    end

    def memory_info
      nil
    end

    def total_memory
      nil
    end

    def create_handle(*args)
      raise "CUDA disabled"
    end

    def destroy_handle(*args)
    end

    def cleanup_handles(*args)
    end

    def gemm(*args)
    end

    def gemm_accumulate(*args)
    end

    def gemm_q8_f32(*args)
      raise "CUDA disabled"
    end

    def gemm_q4_f32(*args)
      raise "CUDA disabled"
    end

    def kv_cache_append_f32(*args)
      raise "CUDA disabled"
    end

    def attention_kv_f32(*args)
      raise "CUDA disabled"
    end

    def kv_cache_append_f16(*args)
      raise "CUDA disabled"
    end

    def attention_kv_f16(*args)
      raise "CUDA disabled"
    end

    def kv_f16_kernels_available? : Bool
      false
    end

    def swiglu_forward(*args)
      raise "CUDA disabled"
    end

    def swiglu_kernel_available? : Bool
      false
    end

    def gather_kernels_available? : Bool
      false
    end

    def rope_forward_rows(*args)
      raise "CUDA disabled"
    end

    def mul_sigmoid(*args)
      raise "CUDA disabled"
    end

    def mul_sigmoid_available?
      false
    end

    def gated_delta_rule(*args)
      raise "CUDA disabled"
    end

    def gated_delta_rule_available?
      false
    end

    def head_rmsnorm_rows(*args)
      raise "CUDA disabled"
    end

    def add_bias_rows(*args)
      raise "CUDA disabled"
    end

    def gemv_q4k(*args)
      raise "CUDA disabled"
    end

    def gemv_q6k(*args)
      raise "CUDA disabled"
    end

    def dequant_q4k_rows(*args)
      raise "CUDA disabled"
    end

    def dequant_q6k_rows(*args)
      raise "CUDA disabled"
    end

    def dequant_k_rows_available? : Bool
      false
    end

    def gdn_gates(*args)
      raise "CUDA disabled"
    end

    def short_conv_silu3(d0 : Pointer(Float32), s0 : Pointer(Float32), st0 : Pointer(Float32), w0 : Pointer(Float32),
                         d1 : Pointer(Float32), s1 : Pointer(Float32), st1 : Pointer(Float32), w1 : Pointer(Float32),
                         d2 : Pointer(Float32), s2 : Pointer(Float32), st2 : Pointer(Float32), w2 : Pointer(Float32),
                         seq : Int32, ch0 : Int32, ch1 : Int32, ch2 : Int32,
                         kernel : Int32, apply_silu : Bool) : Bool
      false
    end

    def short_conv_silu3_available? : Bool
      false
    end

    def short_conv(*args)
      raise "CUDA disabled"
    end

    def gdn_mixer_kernels_available? : Bool
      false
    end

    def gemm_tn(*args)
      raise "CUDA disabled"
    end

    def pack_kv_heads(*args)
      raise "CUDA disabled"
    end

    def prefill_attn_kernels_available? : Bool
      false
    end

    def scatter_add_rows(*args)
      raise "CUDA disabled"
    end

    def rms_norm_forward(*args)
      raise "CUDA disabled"
    end

    def add_inplace(*args)
      raise "CUDA disabled"
    end

    def block_device_kernels_available? : Bool
      false
    end

    def rope_forward(*args)
      raise "CUDA disabled"
    end

    def head_rmsnorm(*args)
      raise "CUDA disabled"
    end

    def attention_device_kernels_available? : Bool
      false
    end

    def geam(*args)
    end

    def scal(*args)
    end

    def ger(*args)
    end

    def dot(*args)
      0.0
    end

    def axpy(*args)
    end

    def softmax_rows(*args)
      raise "CUDA kernels not available"
    end

    def dropout(*args)
      raise "CUDA kernels not available"
    end

    def gather_rows(*args)
      raise "CUDA kernels not available"
    end

    def slice_cols(*args)
      raise "CUDA kernels not available"
    end

    # ameba:disable Naming/AccessorMethodName
    def set_cols(*args)
      raise "CUDA kernels not available"
    end

    def row_mean_var(*args)
      raise "CUDA kernels not available"
    end

    def layer_norm(*args)
      raise "CUDA kernels not available"
    end

    def layer_norm_backward(*args)
      raise "CUDA kernels not available"
    end

    def sum_cols(*args)
      raise "CUDA kernels not available"
    end

    def mul_row_vector(*args)
      raise "CUDA kernels not available"
    end

    def transpose(*args)
      raise "CUDA kernels not available"
    end

    def sigmoid_forward(*args)
      raise "CUDA kernels not available"
    end

    def apply_gradient(*args)
      raise "CUDA kernels not available"
    end

    def accumulate_bias_grad(*args)
      raise "CUDA kernels not available"
    end

    def zero_matrix(*args)
      raise "CUDA kernels not available"
    end

    def fill_matrix(*args)
      raise "CUDA kernels not available"
    end

    def element_div(*args)
      raise "CUDA kernels not available"
    end

    def relu(*args)
      raise "CUDA kernels not available"
    end

    def add_bias(*args)
      raise "CUDA kernels not available"
    end

    def row_sum(*args)
      raise "CUDA kernels not available"
    end

    def count_token_pairs(*args)
      raise "CUDA kernels not available"
    end

    def cross_entropy_loss_gradient(*args) : Int32
      raise "CUDA kernels not available"
    end

    def softmax_cross_entropy_label(*args) : Int32
      raise "CUDA kernels not available"
    end

    def dropout(*args) : Int32
      raise "CUDA kernels not available"
    end

    def relu_backward(*args)
      raise "CUDA kernels not available"
    end

    def softmax_backward(*args)
      raise "CUDA kernels not available"
    end

    def element_log(*args)
      raise "CUDA kernels not available"
    end

    def mse_cost_gradient(*args)
      raise "CUDA kernels not available"
    end
  end

  module CUDNN
    extend self

    def available? : Bool
      false
    end

    def add_bias!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def relu_forward(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def relu_backward(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def sigmoid_forward!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def tanh_forward!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def softmax_rows(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def element_add!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def element_multiply!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def dropout_forward!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def softmax_cross_entropy_loss_and_gradient(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def softmax_cross_entropy_label_loss_and_gradient(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def element_log!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def element_subtract!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def element_addition!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    class CudnnError < Exception
    end

    def check_status(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def handle
      raise CudnnError.new("cuDNN not available")
    end
  end
end
