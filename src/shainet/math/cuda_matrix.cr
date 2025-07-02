require "./simple_matrix"
require "../cuda"
require "./gpu_memory"

module SHAInet
  # Basic GPU matrix wrapper. Allocates device memory when CUDA is
  # available and falls back to SimpleMatrix otherwise. Only matrix
  # multiplication was previously using the GPU.
  class CudaMatrix < SimpleMatrix
    property device_ptr : Pointer(Float64)?
    @device_dirty : Atomic::Flag = Atomic::Flag.new  # Track if GPU data is newer than CPU data

    def initialize(rows : Int32, cols : Int32, init : Float64 = 0.0)
      super(rows, cols, init)
      @device_ptr = Pointer(Float64).null

      # Only allocate GPU memory if CUDA is fully available
      if CUDA.fully_available?
        begin
          ptr = GPUMemory.alloc_buffer(@rows, @cols)
          @device_ptr = ptr unless ptr.null?
        rescue ex
          # If GPU allocation fails, continue with CPU-only mode
          @device_ptr = Pointer(Float64).null
        end
      end
    end

    def finalize
      # Only finalize if we have a valid device pointer and CUDA is still available
      if dptr = @device_ptr
        unless dptr.null?
          begin
            # Double-check CUDA is still available before releasing
            if CUDA.fully_available?
              GPUMemory.release_buffer(dptr, @rows, @cols)
            else
              # If CUDA is no longer available, try direct cleanup
              CUDA.free(dptr.as(Pointer(Void))) rescue nil
            end
          rescue
            # If all cleanup fails, just ignore it - this can happen
            # when CUDA context is no longer available during shutdown
          end
        end
      end
    end

    def self.from_a(array : Array(Array(GenNum)))
      m = new(array.size, array.first.size)
      array.each_with_index do |row, i|
        row.each_with_index do |val, j|
          m.unsafe_set(i, j, val.to_f64)
        end
      end
      m.sync_to_device! if CUDA.available?
      m
    end

    # Return the transposed matrix. When CUDA is available the returned
    # instance keeps the device buffer in sync so that further GPU
    # operations can be used without additional copies.
    def transpose
      result = CudaMatrix.new(@cols, @rows)

      # For GPU matrices, avoid unnecessary CPU access
      if CUDA.available? && device_dirty?
        # Keep the transpose operation minimal and let GPU handle it later
        # For now, sync to CPU and do CPU transpose, but mark result for GPU
        sync_from_device!
      end

      @rows.times do |i|
        @cols.times do |j|
          result.unsafe_set(j, i, unsafe_get(i, j))
        end
      end
      result.sync_to_device! if CUDA.available?
      result
    end

    def sync_to_device!
      return unless dptr = @device_ptr
      return if dptr.null?

      begin
        buf = Array(Float64).new(@rows*@cols, 0.0)
        @rows.times do |i|
          @cols.times do |j|
            buf[i*@cols + j] = self[i, j]  # Row-major order
          end
        end
        bytes = ((@rows*@cols)*8).to_u64
        result = CUDA.memcpy(dptr.as(Pointer(Void)), buf.to_unsafe.as(Pointer(Void)), bytes, CUDA::MemcpyKind::HostToDevice)
        if result != 0
          # Memory copy failed, invalidate device pointer
          @device_ptr = Pointer(Float64).null
        else
          # Mark GPU data as up-to-date
          mark_device_clean!
        end
      rescue
        # If sync fails, invalidate device pointer to prevent future issues
        @device_ptr = Pointer(Float64).null
      end
    end

    def sync_from_device!
      return unless dptr = @device_ptr
      return if dptr.null?
      return unless device_dirty?  # Only sync if GPU data is newer

      begin
        buf = Array(Float64).new(@rows*@cols, 0.0)
        bytes = ((@rows*@cols)*8).to_u64
        result = CUDA.memcpy(buf.to_unsafe.as(Pointer(Void)), dptr.as(Pointer(Void)), bytes, CUDA::MemcpyKind::DeviceToHost)
        if result == 0
          @rows.times do |i|
            @cols.times do |j|
              self[i, j] = buf[i*@cols + j]  # Row-major order
            end
          end
          # Mark CPU data as up-to-date
          mark_device_clean!
        else
          # Memory copy failed, invalidate device pointer
          @device_ptr = Pointer(Float64).null
        end
      rescue
        # If sync fails, invalidate device pointer to prevent future issues
        @device_ptr = Pointer(Float64).null
      end
    end

    def slice_cols(start_col : Int32, length : Int32)
      result = CudaMatrix.new(@rows, length)
      if CUDA.available? && (sptr = self.device_ptr) && (dptr = result.device_ptr) && !sptr.null? && !dptr.null?
        begin
          # Ensure source has up-to-date GPU data
          self.sync_to_device! unless device_dirty?

          CUDA.slice_cols(dptr, sptr, @rows, @cols, start_col, length)

          # Mark result as having newer GPU data
          result.mark_device_dirty!
          return result
        rescue
        end
      end
      @rows.times do |i|
        length.times do |j|
          result.unsafe_set(i, j, self[i, start_col + j])
        end
      end
      result.sync_to_device! if CUDA.available?
      result
    end

    def set_cols!(start_col : Int32, other : CudaMatrix)
      raise ArgumentError.new("row mismatch") unless other.rows == @rows
      if CUDA.available? && (dptr = self.device_ptr) && (sptr = other.device_ptr) && !dptr.null? && !sptr.null?
        begin
          # Ensure both matrices have up-to-date GPU data
          self.sync_to_device! unless device_dirty?
          other.sync_to_device! unless other.device_dirty?

          CUDA.set_cols(dptr, sptr, @rows, @cols, start_col, other.cols)

          # Mark self as having newer GPU data
          mark_device_dirty!
          return self
        rescue
        end
      end
      other.cols.times do |j|
        @rows.times do |i|
          self.unsafe_set(i, start_col + j, other[i, j])
        end
      end
      self.sync_to_device! if CUDA.available?
      self
    end

    def *(other : CudaMatrix)
      if CUDA.available? && (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?
        # Ensure both operands have up-to-date GPU data
        self.sync_to_device! unless device_dirty?
        other.sync_to_device! unless other.device_dirty?

        handle = CUDA.create_handle
        result = CudaMatrix.new(@rows, other.cols)
        CUDA.gemm(handle, ptr_a, ptr_b, result.device_ptr.not_nil!, @rows, other.cols, @cols)
        CUDA.destroy_handle(handle)

        # Mark result as having newer GPU data
        result.mark_device_dirty!
        result
      else
        super(other)
      end
    end

    # Override [] to ensure GPU data is synced to CPU before reading
    # Only sync if we absolutely need to read the value
    def [](row : Int32, col : Int32)
      # If GPU data is newer, sync it to CPU first
      sync_from_device! if device_dirty?
      super(row, col)
    end

    # Override []= to mark CPU data as modified
    def []=(row : Int32, col : Int32, value : Float64)
      super(row, col, value)
      # CPU data is now newer, need to sync to device before next GPU op
      mark_device_clean!
    end

    # Provide a method to access values without syncing (for performance-critical code)
    def unsafe_get(row : Int32, col : Int32)
      @data[row * @cols + col]
    end

    # Provide a method to set values without affecting sync state
    def unsafe_set(row : Int32, col : Int32, value : Float64)
      @data[row * @cols + col] = value
    end

    # Helper methods for device_dirty atomic flag
    def device_dirty?
      # Check if flag is set without modifying it
      # We test_and_set, check the previous value, then restore it if it was clear
      was_set = @device_dirty.test_and_set
      @device_dirty.clear unless was_set
      was_set
    end

    def mark_device_dirty!
      @device_dirty.test_and_set
    end

    def mark_device_clean!
      @device_dirty.clear
    end

  end

  def self.zeros(rows : Int32, cols : Int32)
    m = new(rows, cols)
    # Initial data is already zero, just sync to GPU
    m.sync_to_device! if CUDA.available?
    m
  end

  def self.ones(rows : Int32, cols : Int32)
    m = new(rows, cols)
    rows.times do |i|
      cols.times do |j|
        m.unsafe_set(i, j, 1.0)
      end
    end
    m.sync_to_device! if CUDA.available?
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
    if CUDA.available? && (sptr = self.device_ptr) && (dptr = dup.device_ptr) && !sptr.null? && !dptr.null?
      # If we have GPU data, copy it directly on GPU
      if device_dirty?
        # GPU -> GPU copy
        bytes = (@rows * @cols * 8).to_u64
        result = CUDA.memcpy(dptr.as(Pointer(Void)), sptr.as(Pointer(Void)), bytes, CUDA::MemcpyKind::DeviceToDevice)
        if result == 0
          dup.mark_device_dirty!
          return dup
        end
      end
    end

    # Fallback: CPU copy
    @rows.times do |i|
      @cols.times do |j|
        dup.unsafe_set(i, j, unsafe_get(i, j))
      end
    end
    dup.sync_to_device! if CUDA.available?
    dup
  end

  # In-place element-wise addition.
  def add!(other : CudaMatrix)
    raise ArgumentError.new("size mismatch") unless other.rows == @rows && other.cols == @cols
    if CUDA.available? && (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?
      # Ensure both matrices have up-to-date GPU data
      self.sync_to_device! unless device_dirty?
      other.sync_to_device! unless other.device_dirty?

      handle = CUDA.create_handle
      CUDA.geam(handle, ptr_a, ptr_b, ptr_a, @rows, @cols, 1.0, 1.0)
      CUDA.destroy_handle(handle)

      # Mark self as having newer GPU data
      mark_device_dirty!
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
      # Ensure both operands have up-to-date GPU data
      self.sync_to_device! unless device_dirty?
      other.sync_to_device! unless other.device_dirty?

      handle = CUDA.create_handle
      result = CudaMatrix.new(@rows, @cols)
      CUDA.geam(handle, ptr_a, ptr_b, result.device_ptr.not_nil!, @rows, @cols, 1.0, 1.0)
      CUDA.destroy_handle(handle)

      # Mark result as having newer GPU data
      result.mark_device_dirty!
      result
    else
      result = CudaMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          result.unsafe_set(i, j, self[i, j] + other[i, j])
        end
      end
      result.sync_to_device! if CUDA.available?
      result
    end
  end

  def -(other : CudaMatrix)
    raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
    if CUDA.available? && (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?
      # Ensure both operands have up-to-date GPU data
      self.sync_to_device! unless device_dirty?
      other.sync_to_device! unless other.device_dirty?

      handle = CUDA.create_handle
      result = CudaMatrix.new(@rows, @cols)
      CUDA.geam(handle, ptr_a, ptr_b, result.device_ptr.not_nil!, @rows, @cols, 1.0, -1.0)
      CUDA.destroy_handle(handle)

      # Mark result as having newer GPU data
      result.mark_device_dirty!
      result
    else
      result = CudaMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          result.unsafe_set(i, j, self[i, j] - other[i, j])
        end
      end
      result.sync_to_device! if CUDA.available?
      result
    end
  end

  def *(scalar : Number)
    if CUDA.available? && (dptr = self.device_ptr) && !dptr.null?
      # Ensure self has up-to-date GPU data
      self.sync_to_device! unless device_dirty?

      handle = CUDA.create_handle
      out = self.clone
      ptr = out.device_ptr.not_nil!
      CUDA.scal(handle, ptr, (@rows*@cols), scalar.to_f64)
      CUDA.destroy_handle(handle)

      # Mark result as having newer GPU data
      out.mark_device_dirty!
      out
    else
      result = CudaMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          result.unsafe_set(i, j, self[i, j] * scalar.to_f64)
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
      # Ensure both matrices have up-to-date GPU data
      self.sync_to_device! unless device_dirty?
      bias.sync_to_device! unless bias.device_dirty?

      CUDA.add_bias(dptr, bptr, @rows, @cols)

      # Mark self as having newer GPU data
      mark_device_dirty!
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
      # Ensure self has up-to-date GPU data
      self.sync_to_device! unless device_dirty?

      CUDA.relu(dptr, (@rows*@cols))

      # Mark self as having newer GPU data
      mark_device_dirty!
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
        # Use row-major addressing: each column starts at j*@rows offset in the flattened array
        # But wait - this is wrong for row-major! In row-major, we need to scale each element individually
        # Let's fall back to CPU for now to avoid memory corruption
      end
      CUDA.destroy_handle(handle)
      # Fall back to CPU implementation for safety
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] *= vec[0, j]
        end
      end
      self.sync_to_device! if CUDA.available?
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
