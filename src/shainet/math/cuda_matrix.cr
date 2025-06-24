require "./simple_matrix"
require "../cuda"

module SHAInet
  # Basic GPU matrix wrapper. Allocates device memory when CUDA is
  # available and falls back to SimpleMatrix otherwise. Only matrix
  # multiplication uses the GPU.
  class CudaMatrix < SimpleMatrix
    property device_ptr : Pointer(Float64)?

    def initialize(rows : Int32, cols : Int32)
      super(rows, cols)
      if CUDA.available?
        ptr = Pointer(Float64).null
        bytes = ((@rows*@cols)*8).to_u64
        res = CUDA.malloc(pointerof(ptr).as(Pointer(Pointer(Void))), bytes)
        @device_ptr = res == 0 ? ptr : Pointer(Float64).null
      else
        @device_ptr = Pointer(Float64).null
      end
    end

    def finalize
      if dptr = @device_ptr
        CUDA.free(dptr.as(Pointer(Void))) unless dptr.null?
      end
    end

    def self.from_a(array : Array(Array(GenNum)))
      m = new(array.size, array.first.size)
      array.each_with_index do |row, i|
        row.each_with_index do |val, j|
          m[i, j] = val.to_f64
        end
      end
      m.sync_to_device!
      m
    end

    def sync_to_device!
      return unless dptr = @device_ptr
      return if dptr.null?
      buf = Array(Float64).new(@rows*@cols)
      @rows.times do |i|
        @cols.times do |j|
          buf[j*@rows + i] = self[i, j]
        end
      end
      bytes = ((@rows*@cols)*8).to_u64
      CUDA.memcpy(dptr.as(Pointer(Void)), buf.to_unsafe.as(Pointer(Void)), bytes, CUDA::MemcpyKind::HostToDevice)
    end

    def sync_from_device!
      return unless dptr = @device_ptr
      return if dptr.null?
      buf = Array(Float64).new(@rows*@cols)
      bytes = ((@rows*@cols)*8).to_u64
      CUDA.memcpy(buf.to_unsafe.as(Pointer(Void)), dptr.as(Pointer(Void)), bytes, CUDA::MemcpyKind::DeviceToHost)
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] = buf[j*@rows + i]
        end
      end
    end

    def *(other : CudaMatrix)
      if CUDA.available? && (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?
        handle = CUDA.create_handle
        result = CudaMatrix.new(@rows, other.cols)
        CUDA.gemm(handle, ptr_a, ptr_b, result.device_ptr.not_nil!, @rows, other.cols, @cols)
        CUDA.destroy_handle(handle)
        result.sync_from_device!
        result
      else
        super(other)
      end
    end
  end
end
