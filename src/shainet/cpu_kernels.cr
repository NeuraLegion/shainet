module SHAInet
  # Dynamic binding to libshainet_cpu_kernels.so (AVX2 Q4_K/Q6_K GEMV).
  # Loaded at runtime via dlopen -- optional, Crystal scalar fallback used
  # when the .so is missing or the CPU lacks AVX2/FMA.
  module CPUKernels
    alias GemvProc = Proc(
      Pointer(Float32), Pointer(UInt8), Pointer(Float32),
      Int32, Int32, Int32, Nil,
    )

    @@handle : Pointer(Void)?
    @@gemv_q4k : GemvProc?
    @@gemv_q6k : GemvProc?
    @@checked = false

    def self.available? : Bool
      ensure_loaded
      !!@@handle
    end

    private def self.ensure_loaded
      return if @@checked
      @@checked = true
      handle = LibC.dlopen(
        "libshainet_cpu_kernels.so", LibC::RTLD_LAZY
      )
      return unless handle
      @@handle = handle

      sym = LibC.dlsym(handle, "gemv_q4k_cpu")
      if sym
        @@gemv_q4k = GemvProc.new(
          sym, Pointer(Void).null
        )
      end
      sym = LibC.dlsym(handle, "gemv_q6k_cpu")
      if sym
        @@gemv_q6k = GemvProc.new(
          sym, Pointer(Void).null
        )
      end
    end

    def self.gemv_q4k(
      x : Pointer(Float32), w : Pointer(UInt8),
      y : Pointer(Float32),
      m : Int32, n : Int32, k : Int32,
    ) : Bool
      ensure_loaded
      fn = @@gemv_q4k
      if fn
        fn.call(x, w, y, m, n, k)
        true
      else
        false
      end
    end

    def self.gemv_q6k(
      x : Pointer(Float32), w : Pointer(UInt8),
      y : Pointer(Float32),
      m : Int32, n : Int32, k : Int32,
    ) : Bool
      ensure_loaded
      fn = @@gemv_q6k
      if fn
        fn.call(x, w, y, m, n, k)
        true
      else
        false
      end
    end
  end
end
