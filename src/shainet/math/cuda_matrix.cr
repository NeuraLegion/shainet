require "./simple_matrix"
require "../cuda"

module SHAInet
  # Basic GPU matrix wrapper. Allocates device memory when CUDA is
  # available and falls back to SimpleMatrix otherwise. Only matrix
  # multiplication was previously using the GPU.
  class CudaMatrix < SimpleMatrix
    property device_ptr : Pointer(Float64)?

    def initialize(rows : Int32, cols : Int32, init : Float64 = 0.0)
      super(rows, cols, init)
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

    # Return the transposed matrix. When CUDA is available the returned
    # instance keeps the device buffer in sync so that further GPU
    # operations can be used without additional copies.
    def transpose
      result = CudaMatrix.new(@cols, @rows)
      @rows.times do |i|
        @cols.times do |j|
          result[j, i] = self[i, j]
        end
      end
      result.sync_to_device!
      result
    end

    def sync_to_device!
      return unless dptr = @device_ptr
      return if dptr.null?
      buf = Array(Float64).new(@rows*@cols, 0.0)
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
      buf = Array(Float64).new(@rows*@cols, 0.0)
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

  def self.zeros(rows : Int32, cols : Int32)
    m = new(rows, cols)
    m.sync_to_device!
    m
  end

  def self.ones(rows : Int32, cols : Int32)
    m = new(rows, cols)
    rows.times do |i|
      cols.times do |j|
        m[i, j] = 1.0
      end
    end
    m.sync_to_device!
    m
  end

  def self.tensor(rows : Int32, cols : Int32)
    CudaTensorMatrix.new(rows, cols)
  end

  def random_fill!(min : Float64 = -0.1, max : Float64 = 0.1)
    super(min, max)
    sync_to_device!
    self
  end

  def clone
    dup = CudaMatrix.new(@rows, @cols)
    @rows.times do |i|
      @cols.times do |j|
        dup[i, j] = self[i, j]
      end
    end
    dup.sync_to_device!
    dup
  end

  # In-place element-wise addition.
  def add!(other : CudaMatrix)
    raise ArgumentError.new("size mismatch") unless other.rows == @rows && other.cols == @cols
    if CUDA.available? && (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?
      handle = CUDA.create_handle
      CUDA.geam(handle, ptr_a, ptr_b, ptr_a, @rows, @cols, 1.0, 1.0)
      CUDA.destroy_handle(handle)
      self.sync_from_device!
    else
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] += other[i, j]
        end
      end
      self.sync_to_device! if CUDA.available?
    end
    self
  end

  def +(other : CudaMatrix)
    raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
    if CUDA.available? && (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?
      handle = CUDA.create_handle
      result = CudaMatrix.new(@rows, @cols)
      CUDA.geam(handle, ptr_a, ptr_b, result.device_ptr.not_nil!, @rows, @cols, 1.0, 1.0)
      CUDA.destroy_handle(handle)
      result.sync_from_device!
      result
    else
      result = CudaMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = self[i, j] + other[i, j]
        end
      end
      result.sync_to_device! if CUDA.available?
      result
    end
  end

  def -(other : CudaMatrix)
    raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
    if CUDA.available? && (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?
      handle = CUDA.create_handle
      result = CudaMatrix.new(@rows, @cols)
      CUDA.geam(handle, ptr_a, ptr_b, result.device_ptr.not_nil!, @rows, @cols, 1.0, -1.0)
      CUDA.destroy_handle(handle)
      result.sync_from_device!
      result
    else
      result = CudaMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = self[i, j] - other[i, j]
        end
      end
      result.sync_to_device! if CUDA.available?
      result
    end
  end

  def *(scalar : Number)
    if CUDA.available? && (dptr = self.device_ptr) && !dptr.null?
      handle = CUDA.create_handle
      out = self.clone
      ptr = out.device_ptr.not_nil!
      CUDA.scal(handle, ptr, (@rows*@cols), scalar.to_f64)
      CUDA.destroy_handle(handle)
      out.sync_from_device!
      out
    else
      result = CudaMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = self[i, j] * scalar.to_f64
        end
      end
      result.sync_to_device! if CUDA.available?
      result
    end
  end

  # Add a bias row vector to each row in-place.
  def add_bias!(bias : CudaMatrix)
    raise ArgumentError.new("bias size mismatch") unless bias.rows == 1 && bias.cols == @cols
    if CUDA.available? && (dptr = self.device_ptr) && (bptr = bias.device_ptr) && !dptr.null? && !bptr.null?
      CUDA.add_bias(dptr, bptr, @rows, @cols)
      self.sync_from_device!
    else
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] += bias[0, j]
        end
      end
      self.sync_to_device! if CUDA.available?
    end
    self
  end

  # Element-wise ReLU activation in-place.
  def relu!
    if CUDA.available? && (dptr = self.device_ptr) && !dptr.null?
      CUDA.relu(dptr, (@rows*@cols))
      self.sync_from_device!
    else
      @rows.times do |i|
        @cols.times do |j|
          v = self[i, j]
          self[i, j] = v > 0 ? v : 0.0
        end
      end
      self.sync_to_device! if CUDA.available?
    end
    self
  end

  # Multiply each column by the corresponding value in a row vector in-place.
  def mul_row_vector!(vec : CudaMatrix)
    raise ArgumentError.new("vector size mismatch") unless vec.rows == 1 && vec.cols == @cols
    if CUDA.available? && (dptr = self.device_ptr) && (vptr = vec.device_ptr) && !dptr.null? && !vptr.null?
      handle = CUDA.create_handle
      @cols.times do |j|
        alpha = vec[0, j]
        ptr = dptr + j*@rows
        CUDA.scal(handle, ptr, @rows, alpha)
      end
      CUDA.destroy_handle(handle)
      self.sync_from_device!
    else
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] *= vec[0, j]
        end
      end
      self.sync_to_device! if CUDA.available?
    end
    self
  end
end
