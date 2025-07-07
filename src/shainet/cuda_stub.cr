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

    def malloc_host(*args)
      raise "CUDA disabled"
    end

    def free_host(*args)
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

    def element_mul!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def dropout_forward!(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def softmax_cross_entropy_loss_and_gradient(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def cross_entropy_loss_gradient(*args)
      raise CudnnError.new("cuDNN not available")
    end

    def cross_entropy_loss_and_gradient(*args)
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
