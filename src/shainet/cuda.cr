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
                      beta  : Pointer(Float64), b : Pointer(Float64), ldb : Int32,
                      c : Pointer(Float64), ldc : Int32) : Int32
      fun cublasDscal_v2(handle : Handle, n : Int32,
                         alpha : Pointer(Float64), x : Pointer(Float64), incx : Int32) : Int32
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
             m : Int32, n : Int32, k : Int32)
      alpha = 1.0
      beta = 0.0
      LibCUBLAS.cublasDgemm_v2(handle,
        Operation::N.value, Operation::N.value,
        m, n, k,
        pointerof(alpha), a, m,
        b, k,
        pointerof(beta), c, m)
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
  end
end
