module SHAInet
  # Dynamic binding to libshainet_cpu_kernels.so (AVX2 Q4_K/Q6_K GEMV
  # and fp32 SGEMM). Loaded at runtime via dlopen -- optional, Crystal
  # scalar fallback used when missing.
  module CPUKernels
    alias GemvProc = Proc(
      Pointer(Float32), Pointer(UInt8), Pointer(Float32),
      Int32, Int32, Int32, Nil,
    )
    alias SgemmProc = Proc(
      Pointer(Float32), Pointer(Float32), Pointer(Float32),
      Int32, Int32, Int32, Nil,
    )

    # Scalar reference dequant for one row of an i-quant tensor. Used for the embedding (per-row,
    # on demand) and for transcoding tensors whose type has no fast kernel yet.
    alias DequantRowProc = Proc(Pointer(UInt8), Pointer(Float32), Int32, Nil)
    # Type-dispatched variant: (ggml_type, weights, out, K) -> 1 on success, 0 if unhandled.
    alias DequantAnyProc = Proc(Int32, Pointer(UInt8), Pointer(Float32), Int32, Int32)

    # Parallel generic host GEMV: (ggml_type, weights, x, y, m, n, k) -> 1 on success, 0 if the type
    # has no reference dequant.
    alias GemvAnyProc = Proc(Int32, Pointer(UInt8), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32)

    @@handle : Pointer(Void)?
    @@gemv_q4k : GemvProc?
    @@gemv_q6k : GemvProc?
    @@sgemm : SgemmProc?
    @@dequant_iq4xs_row : DequantRowProc?
    @@dequant_any_row : DequantAnyProc?
    @@gemv_any_host : GemvAnyProc?
    @@checked = false

    # Dequantize K values of one IQ4_XS row into `out`. Returns false when the kernel is missing.
    def self.dequant_iq4xs_row(w : Pointer(UInt8), out_ptr : Pointer(Float32), k : Int32) : Bool
      ensure_loaded
      if fn = @@dequant_iq4xs_row
        fn.call(w, out_ptr, k)
        return true
      end
      false
    end

    # Dequantize K values of one row of ANY supported quantized type into `out`. `ggml_type` is the
    # GGUF type number. Returns false when the type has no reference implementation, which is what
    # lets the loader decide between a device kernel and a load-time transcode.
    def self.dequant_row(ggml_type : Int32, w : Pointer(UInt8), out_ptr : Pointer(Float32), k : Int32) : Bool
      ensure_loaded
      if fn = @@dequant_any_row
        return fn.call(ggml_type, w, out_ptr, k) != 0
      end
      false
    end

    # Generic host GEMV, parallel over output rows. Returns false when the C kernel is unavailable or
    # the type has no reference dequant, so the caller can fall back to the scalar loop.
    def self.gemv_any_host(ggml_type : Int32, w : Pointer(UInt8), x : Pointer(Float32),
                           y : Pointer(Float32), m : Int32, n : Int32, k : Int32) : Bool
      ensure_loaded
      if fn = @@gemv_any_host
        return fn.call(ggml_type, w, x, y, m, n, k) != 0
      end
      false
    end

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

      sym = LibC.dlsym(handle, "gemv_any_host")
      @@gemv_any_host = GemvAnyProc.new(sym, Pointer(Void).null) if sym

      sym = LibC.dlsym(handle, "dequant_any_row")
      @@dequant_any_row = DequantAnyProc.new(sym, Pointer(Void).null) if sym

      sym = LibC.dlsym(handle, "dequant_iq4xs_row")
      @@dequant_iq4xs_row = DequantRowProc.new(sym, Pointer(Void).null) if sym

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
      sym = LibC.dlsym(handle, "sgemm_cpu")
      if sym
        @@sgemm = SgemmProc.new(
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

    # fp32 SGEMM: C[M,N] = A[M,K] * B[K,N], AVX2+OpenMP
    def self.sgemm(
      a : Pointer(Float32), b : Pointer(Float32),
      c : Pointer(Float32),
      m : Int32, n : Int32, k : Int32,
    ) : Bool
      ensure_loaded
      fn = @@sgemm
      if fn
        fn.call(a, b, c, m, n, k)
        true
      else
        false
      end
    end
  end
end
