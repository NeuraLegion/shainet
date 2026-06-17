require "./simple_matrix"
require "./quantized_cuda_matrix" # QuantizedWeight
require "./q4_cuda_matrix"        # Q4CudaMatrix (packing + GPU scratch)
{% if flag?(:enable_cuda) %}
  require "../cuda"
{% else %}
  require "../cuda_stub"
{% end %}

module SHAInet
  # Host-resident Q4 weight matrix for MoE expert offload.
  #
  # Identical Q4 numerics to Q4CudaMatrix (same packing, same kernel — zero
  # quality loss), but the packed 4-bit weights + fp32 scales live in *host*
  # RAM instead of on the GPU. On each `gemv` the weights are uploaded into a
  # small, shape-keyed GPU scratch buffer shared across all offloaded experts,
  # then the standard Q4 kernel runs.
  #
  # This lets a large MoE (e.g. Qwen3-Coder-30B-A3B: ~29B params in experts)
  # keep its experts in cheap host RAM while only the few active experts per
  # token ever touch the GPU, so the model fits a 16GB card with no precision
  # change — trading some PCIe bandwidth for VRAM.
  class Q4HostMatrix
    include QuantizedWeight

    getter rows : Int32 # K (in_features)
    getter cols : Int32 # N (out_features)

    @q_host : Array(UInt8)
    @s_host : Array(Float32)

    # Shared GPU scratch buffers, keyed by [K, N]. Reused across every offloaded
    # expert: experts in a layer share weight shapes, so this holds only a couple
    # of small Q4CudaMatrix buffers regardless of expert count.
    @@scratch = Hash(Tuple(Int32, Int32), Q4CudaMatrix).new

    def initialize(@rows : Int32, @cols : Int32, @q_host : Array(UInt8), @s_host : Array(Float32))
    end

    # Quantize a host fp32 weight [K, N] into Q4 and keep it resident in host RAM.
    def self.from_simple(w : SimpleMatrix) : Q4HostMatrix
      q_host, s_host = Q4CudaMatrix.pack(w)
      new(w.rows, w.cols, q_host, s_host)
    end

    # Host memory footprint in bytes (4-bit weights + fp32 scales).
    def host_bytes : UInt64
      @q_host.size.to_u64 + @s_host.size.to_u64 * 4_u64
    end

    # Resident GPU footprint is ~0 — only the shared scratch (counted once,
    # not per-weight) ever lives on the device.
    def device_bytes : UInt64
      0_u64
    end

    private def scratch : Q4CudaMatrix
      (@@scratch[{@rows, @cols}] ||= Q4CudaMatrix.new(@rows, @cols))
    end

    # Upload this expert's weights into the shared scratch and run the Q4 GEMV.
    def gemv(x : CudaMatrix) : CudaMatrix
      s = scratch
      s.upload(@q_host, @s_host)
      s.gemv(x)
    end

    def gemv_into(x : CudaMatrix, result : CudaMatrix) : CudaMatrix
      s = scratch
      s.upload(@q_host, @s_host)
      s.gemv_into(x, result)
    end
  end
end
