require "./cuda"

# cuDNN bindings for high-performance deep learning operations
@[Link("cudnn")]
lib LibCUDNN
  alias CudnnHandle = Void*
  alias CudnnTensorDescriptor = Void*
  alias CudnnFilterDescriptor = Void*
  alias CudnnConvolutionDescriptor = Void*
  alias CudnnActivationDescriptor = Void*
  alias CudnnPoolingDescriptor = Void*
  alias CudnnDropoutDescriptor = Void*
  alias CudnnRNNDescriptor = Void*
  alias CudnnOpTensorDescriptor = Void*
  alias CudnnAttentionDescriptor = Void*

  # Status codes
  enum CudnnStatus
    CUDNN_STATUS_SUCCESS                      =  0
    CUDNN_STATUS_NOT_INITIALIZED              =  1
    CUDNN_STATUS_ALLOC_FAILED                 =  2
    CUDNN_STATUS_BAD_PARAM                    =  3
    CUDNN_STATUS_INTERNAL_ERROR               =  4
    CUDNN_STATUS_INVALID_VALUE                =  5
    CUDNN_STATUS_ARCH_MISMATCH                =  6
    CUDNN_STATUS_MAPPING_ERROR                =  7
    CUDNN_STATUS_EXECUTION_FAILED             =  8
    CUDNN_STATUS_NOT_SUPPORTED                =  9
    CUDNN_STATUS_LICENSE_ERROR                = 10
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11
    CUDNN_STATUS_RUNTIME_IN_PROGRESS          = 12
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW          = 13
    CUDNN_STATUS_VERSION_MISMATCH             = 14
  end

  # Data types
  enum CudnnDataType
    CUDNN_DATA_FLOAT    =  0
    CUDNN_DATA_DOUBLE   =  1
    CUDNN_DATA_HALF     =  2
    CUDNN_DATA_INT8     =  3
    CUDNN_DATA_INT32    =  4
    CUDNN_DATA_INT8x4   =  5
    CUDNN_DATA_UINT8    =  6
    CUDNN_DATA_UINT8x4  =  7
    CUDNN_DATA_INT8x32  =  8
    CUDNN_DATA_BFLOAT16 =  9
    CUDNN_DATA_INT64    = 10
  end

  # Tensor formats
  enum CudnnTensorFormat
    CUDNN_TENSOR_NCHW        = 0
    CUDNN_TENSOR_NHWC        = 1
    CUDNN_TENSOR_NCHW_VECT_C = 2
  end

  # Activation modes
  enum CudnnActivationMode
    CUDNN_ACTIVATION_SIGMOID      = 0
    CUDNN_ACTIVATION_RELU         = 1
    CUDNN_ACTIVATION_TANH         = 2
    CUDNN_ACTIVATION_CLIPPED_RELU = 3
    CUDNN_ACTIVATION_ELU          = 4
    CUDNN_ACTIVATION_IDENTITY     = 5
    CUDNN_ACTIVATION_SWISH        = 6
  end

  # Math types
  enum CudnnMathType
    CUDNN_DEFAULT_MATH                    = 0
    CUDNN_TENSOR_OP_MATH                  = 1
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2
    CUDNN_FMA_MATH                        = 3
  end

  # Core functions
  fun cudnnCreate(handle : CudnnHandle*) : CudnnStatus
  fun cudnnDestroy(handle : CudnnHandle) : CudnnStatus
  fun cudnnGetVersion : LibC::SizeT
  fun cudnnGetErrorString(status : CudnnStatus) : LibC::Char*

  # Tensor descriptor functions
  fun cudnnCreateTensorDescriptor(tensorDesc : CudnnTensorDescriptor*) : CudnnStatus
  fun cudnnDestroyTensorDescriptor(tensorDesc : CudnnTensorDescriptor) : CudnnStatus
  fun cudnnSetTensorNdDescriptor(tensorDesc : CudnnTensorDescriptor, dataType : CudnnDataType,
                                 nbDims : LibC::Int, dimA : LibC::Int*, strideA : LibC::Int*) : CudnnStatus

  # Activation functions
  fun cudnnCreateActivationDescriptor(activationDesc : CudnnActivationDescriptor*) : CudnnStatus
  fun cudnnDestroyActivationDescriptor(activationDesc : CudnnActivationDescriptor) : CudnnStatus
  fun cudnnSetActivationDescriptor(activationDesc : CudnnActivationDescriptor, mode : CudnnActivationMode,
                                   reluNanOpt : LibC::Int, coef : LibC::Double) : CudnnStatus
  fun cudnnActivationForward(handle : CudnnHandle, activationDesc : CudnnActivationDescriptor,
                             alpha : Void*, xDesc : CudnnTensorDescriptor, x : Void*,
                             beta : Void*, yDesc : CudnnTensorDescriptor, y : Void*) : CudnnStatus
  fun cudnnActivationBackward(handle : CudnnHandle, activationDesc : CudnnActivationDescriptor,
                              alpha : Void*, yDesc : CudnnTensorDescriptor, y : Void*,
                              dyDesc : CudnnTensorDescriptor, dy : Void*,
                              xDesc : CudnnTensorDescriptor, x : Void*,
                              beta : Void*, dxDesc : CudnnTensorDescriptor, dx : Void*) : CudnnStatus

  # OpTensor functions (for element-wise operations like add, multiply) - commented out for compatibility
  # fun cudnnCreateOpTensorDescriptor(opTensorDesc : CudnnOpTensorDescriptor*) : CudnnStatus
  # fun cudnnDestroyOpTensorDescriptor(opTensorDesc : CudnnOpTensorDescriptor) : CudnnStatus
  # fun cudnnSetOpTensorDescriptor(opTensorDesc : CudnnOpTensorDescriptor, opTensorOp : CudnnOpTensorOp,
  #                                opTensorCompType : CudnnDataType, opTensorNanOpt : CudnnNanPropagation) : CudnnStatus
  # fun cudnnOpTensor(handle : CudnnHandle, opTensorDesc : CudnnOpTensorDescriptor,
  #                   alpha1 : Void*, aDesc : CudnnTensorDescriptor, a : Void*,
  #                   alpha2 : Void*, bDesc : CudnnTensorDescriptor, b : Void*,
  #                   beta : Void*, cDesc : CudnnTensorDescriptor, c : Void*) : CudnnStatus

  # Add bias function
  fun cudnnAddTensor(handle : CudnnHandle, alpha : Void*, biasDesc : CudnnTensorDescriptor, bias : Void*,
                     beta : Void*, yDesc : CudnnTensorDescriptor, y : Void*) : CudnnStatus

  # Softmax functions
  enum CudnnSoftmaxAlgorithm
    CUDNN_SOFTMAX_FAST     = 0
    CUDNN_SOFTMAX_ACCURATE = 1
    CUDNN_SOFTMAX_LOG      = 2
  end

  enum CudnnSoftmaxMode
    CUDNN_SOFTMAX_MODE_INSTANCE = 0
    CUDNN_SOFTMAX_MODE_CHANNEL  = 1
  end

  fun cudnnSoftmaxForward(handle : CudnnHandle, algo : CudnnSoftmaxAlgorithm, mode : CudnnSoftmaxMode,
                          alpha : Void*, xDesc : CudnnTensorDescriptor, x : Void*,
                          beta : Void*, yDesc : CudnnTensorDescriptor, y : Void*) : CudnnStatus
  fun cudnnSoftmaxBackward(handle : CudnnHandle, algo : CudnnSoftmaxAlgorithm, mode : CudnnSoftmaxMode,
                           alpha : Void*, yDesc : CudnnTensorDescriptor, y : Void*,
                           dyDesc : CudnnTensorDescriptor, dy : Void*,
                           beta : Void*, dxDesc : CudnnTensorDescriptor, dx : Void*) : CudnnStatus

  # OpTensor operation types - commented out for compatibility
  # enum CudnnOpTensorOp
  #   CUDNN_OP_TENSOR_ADD  = 0
  #   CUDNN_OP_TENSOR_MUL  = 1
  #   CUDNN_OP_TENSOR_MIN  = 2
  #   CUDNN_OP_TENSOR_MAX  = 3
  #   CUDNN_OP_TENSOR_SQRT = 4
  #   CUDNN_OP_TENSOR_NOT  = 5
  # end

  # NaN propagation options - commented out for compatibility
  # enum CudnnNanPropagation
  #   CUDNN_NOT_PROPAGATE_NAN = 0
  #   CUDNN_PROPAGATE_NAN     = 1
  # end
end

module SHAInet
  module CUDNN
    extend self

    @@handle : LibCUDNN::CudnnHandle? = nil
    @@available : Bool? = nil

    class CudnnError < Exception
      def initialize(@status : LibCUDNN::CudnnStatus)
        super("cuDNN error: #{String.new(LibCUDNN.cudnnGetErrorString(@status))}")
      end
    end

    def available?
      @@available ||= begin
        return false unless CUDA.available?

        # Try to create and destroy a cuDNN handle
        result = LibCUDNN.cudnnCreate(out handle)
        if result == LibCUDNN::CudnnStatus::CUDNN_STATUS_SUCCESS
          LibCUDNN.cudnnDestroy(handle)
          Log.info { "cuDNN available, version: #{LibCUDNN.cudnnGetVersion}" }
          true
        else
          Log.info { "cuDNN not available: #{String.new(LibCUDNN.cudnnGetErrorString(result))}" }
          false
        end
      rescue e
        Log.debug { "cuDNN availability check failed: #{e}" }
        false
      end
    end

    def handle
      @@handle ||= begin
        raise "cuDNN not available" unless available?

        result = LibCUDNN.cudnnCreate(out h)
        if result != LibCUDNN::CudnnStatus::CUDNN_STATUS_SUCCESS
          raise CudnnError.new(result)
        end
        h
      end
    end

    def check_status(status : LibCUDNN::CudnnStatus)
      unless status == LibCUDNN::CudnnStatus::CUDNN_STATUS_SUCCESS
        raise CudnnError.new(status)
      end
    end

    # Cleanup
    def cleanup
      if h = @@handle
        LibCUDNN.cudnnDestroy(h)
        @@handle = nil
      end
    end

    # High-level operations

    # Optimized ReLU forward pass
    def self.relu_forward(input : CudaMatrix, output : CudaMatrix)
      raise "Input and output must have same dimensions" unless input.rows == output.rows && input.cols == output.cols

      # Set up 4D tensor descriptors: [batch, channels, height, width] = [rows, cols, 1, 1]
      dims = [input.rows, input.cols, 1, 1]
      strides = [input.cols, 1, 1, 1]

      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out input_desc))
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out output_desc))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        input_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_DOUBLE, 4,
        dims.to_unsafe, strides.to_unsafe))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        output_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_DOUBLE, 4,
        dims.to_unsafe, strides.to_unsafe))

      # Create activation descriptor for ReLU
      CUDNN.check_status(LibCUDNN.cudnnCreateActivationDescriptor(out activation_desc))
      CUDNN.check_status(LibCUDNN.cudnnSetActivationDescriptor(
        activation_desc,
        LibCUDNN::CudnnActivationMode::CUDNN_ACTIVATION_RELU,
        0,  # reluNanOpt
        0.0 # coef
      ))

      alpha = 1.0
      beta = 0.0

      input.sync_to_device!("cudnn_relu_input") unless input.device_dirty?

      # Get device pointers and ensure they're not nil
      input_ptr = input.device_ptr
      output_ptr = output.device_ptr
      raise "Device pointers are nil" if input_ptr.nil? || output_ptr.nil?

      CUDNN.check_status(LibCUDNN.cudnnActivationForward(
        CUDNN.handle,
        activation_desc,
        pointerof(alpha),
        input_desc,
        input_ptr.as(Pointer(Void)),
        pointerof(beta),
        output_desc,
        output_ptr.as(Pointer(Void))
      ))

      output.mark_device_dirty!

      # Clean up
      LibCUDNN.cudnnDestroyActivationDescriptor(activation_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(input_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(output_desc)
    end

    # Optimized ReLU backward pass
    def self.relu_backward(input : CudaMatrix, grad_output : CudaMatrix, grad_input : CudaMatrix)
      raise "Matrices must have same dimensions" unless input.rows == grad_output.rows && input.rows == grad_input.rows

      # Set up 4D tensor descriptors
      dims = [input.rows, input.cols, 1, 1]
      strides = [input.cols, 1, 1, 1]

      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out input_desc))
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out grad_desc))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        input_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_DOUBLE, 4,
        dims.to_unsafe, strides.to_unsafe))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        grad_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_DOUBLE, 4,
        dims.to_unsafe, strides.to_unsafe))

      # Create activation descriptor for ReLU
      CUDNN.check_status(LibCUDNN.cudnnCreateActivationDescriptor(out activation_desc))
      CUDNN.check_status(LibCUDNN.cudnnSetActivationDescriptor(
        activation_desc,
        LibCUDNN::CudnnActivationMode::CUDNN_ACTIVATION_RELU,
        0,  # reluNanOpt
        0.0 # coef
      ))

      alpha = 1.0
      beta = 0.0

      input.sync_to_device!("cudnn_relu_backward_input") unless input.device_dirty?
      grad_output.sync_to_device!("cudnn_relu_backward_grad") unless grad_output.device_dirty?

      # Get device pointers and ensure they're not nil
      input_ptr = input.device_ptr
      grad_output_ptr = grad_output.device_ptr
      grad_input_ptr = grad_input.device_ptr
      raise "Device pointers are nil" if input_ptr.nil? || grad_output_ptr.nil? || grad_input_ptr.nil?

      CUDNN.check_status(LibCUDNN.cudnnActivationBackward(
        CUDNN.handle,
        activation_desc,
        pointerof(alpha),
        input_desc,
        input_ptr.as(Pointer(Void)),
        grad_desc,
        grad_output_ptr.as(Pointer(Void)),
        input_desc,
        input_ptr.as(Pointer(Void)),
        pointerof(beta),
        grad_desc,
        grad_input_ptr.as(Pointer(Void))
      ))

      grad_input.mark_device_dirty!

      # Clean up
      LibCUDNN.cudnnDestroyActivationDescriptor(activation_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(input_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(grad_desc)
    end

    # Optimized bias addition
    def self.add_bias!(matrix : CudaMatrix, bias : CudaMatrix)
      raise "Bias must be a row vector" unless bias.rows == 1 && bias.cols == matrix.cols

      # For bias addition, cuDNN expects the bias to be a 4D tensor with shape [1, C, 1, 1]
      # and the matrix to be [N, C, H, W]. For 2D matrices, we treat them as [N, C, 1, 1]

      # Matrix descriptor: treat as [batch_size, channels, 1, 1]
      matrix_dims = [matrix.rows, matrix.cols, 1, 1]
      matrix_strides = [matrix.cols, 1, 1, 1]

      # Bias descriptor: [1, channels, 1, 1]
      bias_dims = [1, bias.cols, 1, 1]
      bias_strides = [bias.cols, 1, 1, 1]

      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out matrix_desc))
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out bias_desc))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        matrix_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_DOUBLE, 4,
        matrix_dims.to_unsafe, matrix_strides.to_unsafe))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        bias_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_DOUBLE, 4,
        bias_dims.to_unsafe, bias_strides.to_unsafe))

      alpha = 1.0
      beta = 1.0 # Add to existing values

      matrix.sync_to_device!("cudnn_bias_matrix") unless matrix.device_dirty?
      bias.sync_to_device!("cudnn_bias_vector") unless bias.device_dirty?

      # Get device pointers and ensure they're not nil
      bias_ptr = bias.device_ptr
      matrix_ptr = matrix.device_ptr
      raise "Device pointers are nil" if bias_ptr.nil? || matrix_ptr.nil?

      CUDNN.check_status(LibCUDNN.cudnnAddTensor(
        CUDNN.handle,
        pointerof(alpha),
        bias_desc,
        bias_ptr.as(Pointer(Void)),
        pointerof(beta),
        matrix_desc,
        matrix_ptr.as(Pointer(Void))
      ))

      matrix.mark_device_dirty!

      # Clean up
      LibCUDNN.cudnnDestroyTensorDescriptor(matrix_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(bias_desc)
    end

    # Optimized softmax (for attention) with proper descriptor management
    def self.softmax_rows(input : CudaMatrix, output : CudaMatrix)
      raise "Input and output must have same dimensions" unless input.rows == output.rows && input.cols == output.cols

      # Set up 4D tensor descriptors: [batch, channels, height, width] = [rows, cols, 1, 1]
      dims = [input.rows, input.cols, 1, 1]
      strides = [input.cols, 1, 1, 1]

      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out input_desc))
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out output_desc))

      begin
        CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
          input_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_DOUBLE, 4,
          dims.to_unsafe, strides.to_unsafe))

        CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
          output_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_DOUBLE, 4,
          dims.to_unsafe, strides.to_unsafe))

        alpha = 1.0
        beta = 0.0

        input.sync_to_device!("cudnn_softmax_input") unless input.device_dirty?

        # Get device pointers and ensure they're not nil
        input_ptr = input.device_ptr
        output_ptr = output.device_ptr
        raise "Device pointers are nil" if input_ptr.nil? || output_ptr.nil?

        CUDNN.check_status(LibCUDNN.cudnnSoftmaxForward(
          CUDNN.handle,
          LibCUDNN::CudnnSoftmaxAlgorithm::CUDNN_SOFTMAX_ACCURATE,
          LibCUDNN::CudnnSoftmaxMode::CUDNN_SOFTMAX_MODE_INSTANCE,
          pointerof(alpha),
          input_desc,
          input_ptr.as(Pointer(Void)),
          pointerof(beta),
          output_desc,
          output_ptr.as(Pointer(Void))
        ))

        output.mark_device_dirty!
      ensure
        # Clean up descriptors
        LibCUDNN.cudnnDestroyTensorDescriptor(input_desc)
        LibCUDNN.cudnnDestroyTensorDescriptor(output_desc)
      end
    end

    # Optimized element-wise addition using cuDNN (fallback to cuBLAS if OpTensor not available)
    def self.element_add!(result : CudaMatrix, a : CudaMatrix, b : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = 1.0)
      # For now, fall back to cuBLAS GEAM since OpTensor may not be available in all cuDNN versions
      # This can be optimized later when OpTensor support is confirmed
      raise "Matrices must have same dimensions" unless a.rows == b.rows && a.cols == b.cols && result.rows == a.rows && result.cols == a.cols

      a.sync_to_device!("cudnn_element_add_a") unless a.device_dirty?
      b.sync_to_device!("cudnn_element_add_b") unless b.device_dirty?

      # Use cuBLAS GEAM for element-wise addition as fallback
      handle = CUDA.create_handle
      begin
        CUDA.geam(handle, a.device_ptr.not_nil!, b.device_ptr.not_nil!,
          result.device_ptr.not_nil!, a.rows, a.cols, alpha, beta)
      ensure
        CUDA.destroy_handle(handle)
      end

      result.mark_device_dirty!
    end

    # Optimized element-wise multiplication (fallback to custom kernel)
    def self.element_mul!(result : CudaMatrix, a : CudaMatrix, b : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = 0.0)
      # Element-wise multiplication requires custom implementation since cuBLAS doesn't have direct support
      # For now, fall back to CPU implementation
      raise "Matrices must have same dimensions" unless a.rows == b.rows && a.cols == b.cols && result.rows == a.rows && result.cols == a.cols

      # Sync to CPU for element-wise operations
      a.sync_from_device!("cudnn_element_mul_fallback") if a.device_dirty?
      b.sync_from_device!("cudnn_element_mul_fallback") if b.device_dirty?

      a.rows.times do |i|
        a.cols.times do |j|
          a_val = a.unsafe_get(i, j)
          b_val = b.unsafe_get(i, j)
          result.unsafe_set(i, j, alpha * a_val * b_val + beta * (result.device_dirty? ? 0.0 : result.unsafe_get(i, j)))
        end
      end

      result.sync_to_device!("cudnn_element_mul_result")
    end
  end
end
