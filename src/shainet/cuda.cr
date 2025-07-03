require "log"

module SHAInet
  module CUDA
    Log = ::Log.for(self)
    extend self

    # :nodoc:
    @[Link("cudart", ldflags: "-L/usr/local/cuda/targets/x86_64-linux/lib")]
    lib LibCUDARuntime
      fun cudaRuntimeGetVersion(version : Pointer(Int32)) : Int32
      fun cudaMalloc(ptr : Pointer(Pointer(Void)), size : LibC::SizeT) : Int32
      fun cudaFree(ptr : Pointer(Void)) : Int32
      fun cudaMemcpy(dst : Pointer(Void), src : Pointer(Void), count : LibC::SizeT, kind : Int32) : Int32
    end

    @[Link("cublas", ldflags: "-L/usr/local/cuda/targets/x86_64-linux/lib")]
    lib LibCUBLAS
      type Handle = Void*

      fun cublasCreate_v2(handle : Pointer(Handle)) : Int32
      fun cublasDestroy_v2(handle : Handle) : Int32
      fun cublasDgemm_v2(handle : Handle, transa : Int32, transb : Int32,
                         m : Int32, n : Int32, k : Int32,
                         alpha : Pointer(Float64), a : Pointer(Float64), lda : Int32,
                         b : Pointer(Float64), ldb : Int32,
                         beta : Pointer(Float64), c : Pointer(Float64), ldc : Int32) : Int32
      fun cublasDgeam(handle : Handle,
                      transa : Int32, transb : Int32,
                      m : Int32, n : Int32,
                      alpha : Pointer(Float64), a : Pointer(Float64), lda : Int32,
                      beta : Pointer(Float64), b : Pointer(Float64), ldb : Int32,
                      c : Pointer(Float64), ldc : Int32) : Int32
      fun cublasDscal_v2(handle : Handle, n : Int32,
                         alpha : Pointer(Float64), x : Pointer(Float64), incx : Int32) : Int32
      fun cublasDger_v2(handle : Handle,
                        m : Int32, n : Int32,
                        alpha : Pointer(Float64),
                        x : Pointer(Float64), incx : Int32,
                        y : Pointer(Float64), incy : Int32,
                        a : Pointer(Float64), lda : Int32) : Int32
      fun cublasDdot_v2(handle : Handle, n : Int32,
                        x : Pointer(Float64), incx : Int32,
                        y : Pointer(Float64), incy : Int32,
                        result : Pointer(Float64)) : Int32
      fun cublasDaxpy_v2(handle : Handle, n : Int32,
                         alpha : Pointer(Float64),
                         x : Pointer(Float64), incx : Int32,
                         y : Pointer(Float64), incy : Int32) : Int32
    end

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

    # Check if CUDA runtime and cuBLAS libraries can be opened.
    @@checked = false
    @@available = false

    def available?
      return false if ENV["SHAINET_DISABLE_CUDA"]?
      return @@available if @@checked
      @@checked = true

      rt = LibC.dlopen("libcudart.so", LibC::RTLD_LAZY)
      if rt.null?
        err = LibC.dlerror
        msg = err.null? ? "unknown" : String.new(err)
        Log.debug { "Failed to load libcudart.so: #{msg}. LD_LIBRARY_PATH=#{ENV["LD_LIBRARY_PATH"]?}" }
      end

      blas = LibC.dlopen("libcublas.so", LibC::RTLD_LAZY)
      if blas.null?
        err = LibC.dlerror
        msg = err.null? ? "unknown" : String.new(err)
        Log.debug { "Failed to load libcublas.so: #{msg}. LD_LIBRARY_PATH=#{ENV["LD_LIBRARY_PATH"]?}" }
      end

      if rt.null? || blas.null?
        @@available = false
      else
        LibC.dlclose(rt)
        LibC.dlclose(blas)
        @@available = true
      end

      @@available
    rescue e
      Log.error { "CUDA availability check raised: #{e}" }
      @@available = false
    end

    # Returns the CUDA runtime version or nil if CUDA is unavailable.
    def version
      return nil unless available?
      out = 0
      if LibCUDARuntime.cudaRuntimeGetVersion(pointerof(out)) == 0
        out
      else
        nil
      end
    rescue
      nil
    end

    # Returns true when the cuDNN library can be loaded.
    def cudnn_available?
      handle = LibC.dlopen("libcudnn.so", LibC::RTLD_LAZY)
      if handle.null?
        err = LibC.dlerror
        msg = err.null? ? "unknown" : String.new(err)
        Log.debug { "Failed to load libcudnn.so: #{msg}. LD_LIBRARY_PATH=#{ENV["LD_LIBRARY_PATH"]?}" }
        false
      else
        LibC.dlclose(handle)
        true
      end
    rescue e
      Log.error { "cuDNN availability check raised: #{e}" }
      false
    end

    # Check if optional CUDA kernels are available via libshainet_cuda_kernels.so
    def kernels_available?
      handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
      if handle.null?
        err = LibC.dlerror
        msg = err.null? ? "unknown" : String.new(err)
        Log.debug { "Failed to load libshainet_cuda_kernels.so: #{msg}. LD_LIBRARY_PATH=#{ENV["LD_LIBRARY_PATH"]?}" }
        false
      else
        LibC.dlclose(handle)
        true
      end
    rescue e
      Log.error { "kernel availability check raised: #{e}" }
      false
    end

    def malloc(ptr : Pointer(Pointer(Void)), size : LibC::SizeT)
      LibCUDARuntime.cudaMalloc(ptr, size)
    end

    def free(ptr : Pointer(Void))
      LibCUDARuntime.cudaFree(ptr)
    end

    def memcpy(dst : Pointer(Void), src : Pointer(Void), bytes : LibC::SizeT, kind : MemcpyKind)
      LibCUDARuntime.cudaMemcpy(dst, src, bytes, kind.value)
    end

    def create_handle
      handle = Pointer(LibCUBLAS::Handle).malloc(1)
      raise "cublasCreate failed" unless LibCUBLAS.cublasCreate_v2(handle) == 0
      handle.value
    end

    def destroy_handle(handle : LibCUBLAS::Handle)
      LibCUBLAS.cublasDestroy_v2(handle)
    end

    def gemm(handle : LibCUBLAS::Handle, a : Pointer(Float64), b : Pointer(Float64), c : Pointer(Float64),
             m : Int32, n : Int32, k : Int32, lda : Int32, ldb : Int32, ldc : Int32)
      alpha = 1.0
      beta = 0.0
      LibCUBLAS.cublasDgemm_v2(handle,
        Operation::N.value, Operation::N.value,
        m, n, k,
        pointerof(alpha), a, lda,
        b, ldb,
        pointerof(beta), c, ldc)
    end

    def geam(handle : LibCUBLAS::Handle, a : Pointer(Float64), b : Pointer(Float64), c : Pointer(Float64),
             m : Int32, n : Int32, alpha : Float64, beta : Float64)
      LibCUBLAS.cublasDgeam(handle,
        Operation::N.value, Operation::N.value,
        m, n,
        pointerof(alpha), a, m,
        pointerof(beta), b, m,
        c, m)
    end

    def scal(handle : LibCUBLAS::Handle, x : Pointer(Float64), n : Int32, alpha : Float64)
      LibCUBLAS.cublasDscal_v2(handle, n, pointerof(alpha), x, 1)
    end

    def ger(handle : LibCUBLAS::Handle, x : Pointer(Float64), y : Pointer(Float64), a : Pointer(Float64), m : Int32, n : Int32, lda : Int32, alpha : Float64 = 1.0)
      LibCUBLAS.cublasDger_v2(handle, m, n, pointerof(alpha), x, 1, y, 1, a, lda)
    end

    def dot(handle : LibCUBLAS::Handle, x : Pointer(Float64), y : Pointer(Float64), n : Int32)
      result = 0.0
      LibCUBLAS.cublasDdot_v2(handle, n, x, 1, y, 1, pointerof(result))
      result
    end

    def axpy(handle : LibCUBLAS::Handle, alpha : Float64, x : Pointer(Float64), y : Pointer(Float64), n : Int32)
      LibCUBLAS.cublasDaxpy_v2(handle, n, pointerof(alpha), x, 1, y, 1)
      # Optional kernels implemented in src/shainet/native/cuda_kernels.cu
      # These methods fall back to CPU when the native library is missing.
    end

    # Optional kernels implemented in src/shainet/native/cuda_kernels.cu
    # These methods dynamically load from libshainet_cuda_kernels.so when available
    @@softmax_rows_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@dropout_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Float64, UInt64, Void)? = nil
    @@gather_rows_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Int32), Int32, Int32, Void)? = nil
    @@slice_cols_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Int32, Int32, Void)? = nil
    @@set_cols_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Int32, Int32, Void)? = nil
    @@row_mean_var_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@layer_norm_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Float64, Void)? = nil
    @@layer_norm_backward_proc : Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Float64, Void)? = nil
    @@sum_cols_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil
    @@mul_row_vector_proc : Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void)? = nil

    def softmax_rows(dst : Pointer(Float64), src : Pointer(Float64), rows : Int32, cols : Int32)
      unless fn = @@softmax_rows_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "softmax_rows")
          unless sym.null?
            @@softmax_rows_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@softmax_rows_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, cols)
    end

    def dropout(dst : Pointer(Float64), src : Pointer(Float64), rows : Int32, cols : Int32, drop_p : Float64, seed : UInt64)
      unless fn = @@dropout_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "dropout")
          unless sym.null?
            @@dropout_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Float64, UInt64, Void).new(sym, Pointer(Void).null)
            fn = @@dropout_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, cols, drop_p, seed)
    end

    def gather_rows(dst : Pointer(Float64), src : Pointer(Float64), ids : Pointer(Int32), rows : Int32, cols : Int32)
      unless fn = @@gather_rows_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "gather_rows")
          unless sym.null?
            @@gather_rows_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Int32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@gather_rows_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, ids, rows, cols)
    end

    def slice_cols(dst : Pointer(Float64), src : Pointer(Float64), rows : Int32, src_cols : Int32, start_col : Int32, len : Int32)
      unless fn = @@slice_cols_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "slice_cols")
          unless sym.null?
            @@slice_cols_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@slice_cols_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, src_cols, start_col, len)
    end

    def set_cols(dst : Pointer(Float64), src : Pointer(Float64), rows : Int32, dst_cols : Int32, start_col : Int32, len : Int32)
      unless fn = @@set_cols_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "set_cols")
          unless sym.null?
            @@set_cols_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@set_cols_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, dst_cols, start_col, len)
    end

    def row_mean_var(src : Pointer(Float64), mean : Pointer(Float64), var : Pointer(Float64), rows : Int32, cols : Int32)
      unless fn = @@row_mean_var_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "row_mean_var")
          unless sym.null?
            @@row_mean_var_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@row_mean_var_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(src, mean, var, rows, cols)
    end

    def layer_norm(dst : Pointer(Float64), src : Pointer(Float64), mean : Pointer(Float64), var : Pointer(Float64), rows : Int32, cols : Int32, eps : Float64)
      unless fn = @@layer_norm_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "apply_layer_norm") # Note: the actual function name is apply_layer_norm
          unless sym.null?
            @@layer_norm_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Float64, Void).new(sym, Pointer(Void).null)
            fn = @@layer_norm_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, mean, var, rows, cols, eps)
    end

    def layer_norm_backward(d_x : Pointer(Float64), d_gamma : Pointer(Float64), d_beta : Pointer(Float64),
                            d_out : Pointer(Float64), x : Pointer(Float64), gamma : Pointer(Float64),
                            mean : Pointer(Float64), var : Pointer(Float64), norm : Pointer(Float64),
                            rows : Int32, cols : Int32, eps : Float64)
      unless fn = @@layer_norm_backward_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "layer_norm_backward")
          unless sym.null?
            @@layer_norm_backward_proc = Proc(Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Pointer(Float64), Int32, Int32, Float64, Void).new(sym, Pointer(Void).null)
            fn = @@layer_norm_backward_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(d_x, d_gamma, d_beta, d_out, x, gamma, mean, var, norm, rows, cols, eps)
    end

    def sum_cols(dst : Pointer(Float64), src : Pointer(Float64), rows : Int32, cols : Int32)
      unless fn = @@sum_cols_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "sum_cols")
          unless sym.null?
            @@sum_cols_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@sum_cols_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, cols)
    end

    def mul_row_vector(matrix : Pointer(Float64), vec : Pointer(Float64), rows : Int32, cols : Int32)
      unless fn = @@mul_row_vector_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "mul_row_vector")
          unless sym.null?
            @@mul_row_vector_proc = Proc(Pointer(Float64), Pointer(Float64), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@mul_row_vector_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(matrix, vec, rows, cols)
    end

    # In-place element-wise ReLU on GPU memory. This fallback implementation
    # copies the data to the host, applies ReLU and writes the result back. It
    # avoids additional synchronization logic in the caller while still keeping
    # the computation on the GPU when proper kernels are available.
    def relu(ptr : Pointer(Float64), len : Int32)
      host = Array(Float64).new(len, 0.0)
      bytes = (len * 8).to_u64
      memcpy(host.to_unsafe.as(Pointer(Void)), ptr.as(Pointer(Void)), bytes, MemcpyKind::DeviceToHost)
      len.times do |i|
        v = host[i]
        host[i] = v > 0 ? v : 0.0
      end
      memcpy(ptr.as(Pointer(Void)), host.to_unsafe.as(Pointer(Void)), bytes, MemcpyKind::HostToDevice)
    end

    # Add a bias row vector to each row of a matrix in GPU memory. Uses the
    # cuBLAS GER routine for the rank-1 update.
    def add_bias(mat : Pointer(Float64), bias : Pointer(Float64), rows : Int32, cols : Int32)
      handle = create_handle
      ones_host = Array(Float64).new(rows, 1.0)
      ones_dev = Pointer(Float64).null
      bytes = (rows * 8).to_u64
      malloc(pointerof(ones_dev).as(Pointer(Pointer(Void))), bytes)
      memcpy(ones_dev.as(Pointer(Void)), ones_host.to_unsafe.as(Pointer(Void)), bytes, MemcpyKind::HostToDevice)
      ger(handle, ones_dev, bias, mat, rows, cols)
      destroy_handle(handle)
      free(ones_dev.as(Pointer(Void)))
    end

    # Accumulate the sum over rows of a matrix into an existing row vector.
    # Performs: dst += ones^T * src
    def row_sum(dst : Pointer(Float64), src : Pointer(Float64), rows : Int32, cols : Int32)
      handle = create_handle
      ones_host = Array(Float64).new(rows, 1.0)
      ones_dev = Pointer(Float64).null
      bytes = (rows * 8).to_u64
      malloc(pointerof(ones_dev).as(Pointer(Pointer(Void))), bytes)
      memcpy(ones_dev.as(Pointer(Void)), ones_host.to_unsafe.as(Pointer(Void)), bytes, MemcpyKind::HostToDevice)
      alpha = 1.0
      beta = 1.0
      LibCUBLAS.cublasDgemm_v2(handle,
        Operation::N.value, Operation::N.value,
        1, cols, rows,
        pointerof(alpha), ones_dev, 1,
        src, rows,
        pointerof(beta), dst, 1)
      destroy_handle(handle)
      free(ones_dev.as(Pointer(Void)))
    end

    # Count token pairs using a custom CUDA kernel when available.
    @@count_pairs_proc : Proc(Pointer(Int32), Pointer(Int32), Pointer(Int32), Pointer(Int32), Int32, Int32, Void)? = nil
    @@kernels_handle : Pointer(Void) = Pointer(Void).null

    def count_token_pairs(counts : Pointer(Int32), a : Pointer(Int32), b : Pointer(Int32), freqs : Pointer(Int32), pair_count : Int32, vocab : Int32)
      unless fn = @@count_pairs_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "count_token_pairs")
          unless sym.null?
            @@count_pairs_proc = Proc(Pointer(Int32), Pointer(Int32), Pointer(Int32), Pointer(Int32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@count_pairs_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(counts, a, b, freqs, pair_count, vocab)
    end

    # Check if both CUDA runtime and custom kernels are available
    def fully_available?
      available? && kernels_available?
    end
  end
end
