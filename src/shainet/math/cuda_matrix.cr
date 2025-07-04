require "./simple_matrix"
require "../cuda"

module SHAInet
  # Basic GPU matrix wrapper. Allocates device memory when CUDA is
  # available. This class is standalone and doesn't inherit from SimpleMatrix
  # to avoid method resolution conflicts.
  class CudaMatrix
    property device_ptr : Pointer(Float64)?
    @device_dirty : Bool = false # Track if GPU data is newer than CPU data
    @rows : Int32
    @cols : Int32
    @data : Array(Float64)
    @gpu_memory_size : UInt64 = 0_u64 # Track our own GPU memory size

    # Global GPU memory tracking
    @@total_gpu_memory_allocated = 0_u64
    @@active_matrices = 0
    @@max_gpu_memory = 16_000_000_000_u64 # 16GB limit (use most of available GPU memory)
    @@allocation_attempts = 0
    @@allocation_failures = 0

    getter rows, cols

    def self.gpu_memory_stats
      {
        active_matrices:       @@active_matrices,
        total_allocated_bytes: @@total_gpu_memory_allocated,
        max_allowed_bytes:     @@max_gpu_memory,
        total_attempts:        @@allocation_attempts,
        allocation_failures:   @@allocation_failures,
      }
    end

    def self.print_detailed_stats
      Log.info { "GPU Memory Statistics:" }
      Log.info { "  Total attempts: #{@@allocation_attempts}" }
      Log.info { "  Failed attempts: #{@@allocation_failures}" }
      Log.info { "  Success rate: #{@@allocation_attempts > 0 ? (100.0 * (@@allocation_attempts - @@allocation_failures) / @@allocation_attempts).round(2) : 0}%" }
      Log.info { "  Active matrices: #{@@active_matrices}" }
      Log.info { "  Total GPU memory: #{@@total_gpu_memory_allocated} bytes (#{(@@total_gpu_memory_allocated / 1024.0 / 1024.0).round(2)} MB)" }
      Log.info { "  Memory limit: #{@@max_gpu_memory} bytes (#{(@@max_gpu_memory / 1024.0 / 1024.0).round(2)} MB)" }
      Log.info { "  Usage %: #{(100.0 * @@total_gpu_memory_allocated / @@max_gpu_memory).round(2)}%" }
      Log.info { "  Average size per matrix: #{@@active_matrices > 0 ? (@@total_gpu_memory_allocated / @@active_matrices).round(2) : 0} bytes" }
    end

    def self.force_cleanup_all
      old_count = @@active_matrices
      old_bytes = @@total_gpu_memory_allocated

      # Multiple rounds of aggressive garbage collection
      5.times do
        GC.collect
        Fiber.yield # Allow finalizers to run
      end

      Log.info { "GPU Memory cleanup: #{old_count} -> #{@@active_matrices} matrices, #{old_bytes} -> #{@@total_gpu_memory_allocated} bytes (freed #{old_bytes - @@total_gpu_memory_allocated} bytes)" }
    end

    def initialize(@rows : Int32, @cols : Int32, init : Float64 = 0.0)
      @data = Array(Float64).new(@rows * @cols, init)
      @device_ptr = Pointer(Float64).null

      # Each CudaMatrix manages its own GPU memory directly
      if CUDA.fully_available?
        size = @rows * @cols
        bytes = (size * 8).to_u64

        # Check if we would exceed memory limits or are getting close
        if @@total_gpu_memory_allocated + bytes > @@max_gpu_memory ||
           @@total_gpu_memory_allocated > (@@max_gpu_memory * 0.8).to_u64 # 80% threshold
          Log.warn { "CudaMatrix.initialize: GPU memory usage high (#{@@total_gpu_memory_allocated}/#{@@max_gpu_memory} bytes, #{@@active_matrices} matrices). Forcing cleanup..." }
          # Force very aggressive cleanup
          3.times { GC.collect }
          # Try again after cleanup
          if @@total_gpu_memory_allocated + bytes > @@max_gpu_memory
            Log.warn { "CudaMatrix.initialize: Still would exceed limit after cleanup. Using CPU-only mode for #{@rows}x#{@cols}" }
            return
          end
        end

        Log.debug { "CudaMatrix.initialize: Attempting direct GPU memory allocation for #{@rows}x#{@cols} matrix (#{bytes} bytes). Current usage: #{@@active_matrices} matrices, #{@@total_gpu_memory_allocated} bytes" }
        @@allocation_attempts += 1
        begin
          ptr = Pointer(Float64).null
          result = CUDA.malloc(pointerof(ptr).as(Pointer(Pointer(Void))), bytes)

          if result == 0 && !ptr.null?
            @device_ptr = ptr
            @gpu_memory_size = bytes
            @@total_gpu_memory_allocated += bytes
            @@active_matrices += 1
            Log.debug { "CudaMatrix.initialize: Successfully allocated #{bytes} bytes GPU memory at #{ptr.address} for #{@rows}x#{@cols}. Total: #{@@active_matrices} matrices, #{@@total_gpu_memory_allocated} bytes" }
          else
            @@allocation_failures += 1
            Log.warn { "CudaMatrix.initialize: Direct GPU allocation failed with result #{result} for #{@rows}x#{@cols}. Total usage: #{@@active_matrices} matrices, #{@@total_gpu_memory_allocated} bytes" }
            @device_ptr = Pointer(Float64).null
            @gpu_memory_size = 0_u64
          end
        rescue ex
          Log.error { "CudaMatrix.initialize: GPU allocation exception for #{@rows}x#{@cols}: #{ex}" }
          @device_ptr = Pointer(Float64).null
          @gpu_memory_size = 0_u64
        end
      else
        Log.debug { "CudaMatrix.initialize: CUDA not available, using CPU-only mode for #{@rows}x#{@cols}" }
      end
    end

    # Basic matrix access operations
    def [](row : Int32, col : Int32)
      # If GPU data is newer, sync it to CPU first
      sync_from_device! if device_dirty?
      @data[row * @cols + col]
    end

    def []=(row : Int32, col : Int32, value : Float64)
      @data[row * @cols + col] = value
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

    def self.from_a(array : Array(Array(GenNum)))
      m = new(array.size, array.first.size)
      array.each_with_index do |row, i|
        row.each_with_index do |val, j|
          m.unsafe_set(i, j, val.to_f64)
        end
      end
      m.sync_to_device! if CUDA.fully_available?
      m
    end

    def self.zeros(rows : Int32, cols : Int32)
      m = new(rows, cols)
      # Initial data is already zero, just sync to GPU
      m.sync_to_device! if CUDA.fully_available?
      m
    end

    def self.ones(rows : Int32, cols : Int32)
      m = new(rows, cols)
      rows.times do |i|
        cols.times do |j|
          m.unsafe_set(i, j, 1.0)
        end
      end
      m.sync_to_device! if CUDA.fully_available?
      m
    end

    def random_fill!(min : Float64 = -0.1, max : Float64 = 0.1)
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] = Random.rand(min..max)
        end
      end
      sync_to_device!
      self
    end

    def finalize
      # Each CudaMatrix cleans up its own GPU memory directly
      if dptr = @device_ptr
        unless dptr.null?
          begin
            Log.debug { "CudaMatrix.finalize: Freeing #{@gpu_memory_size} bytes GPU memory at #{dptr.address} for #{@rows}x#{@cols}" }
            CUDA.free(dptr.as(Pointer(Void)))
            @@total_gpu_memory_allocated -= @gpu_memory_size
            @@active_matrices -= 1
            @device_ptr = Pointer(Float64).null
            @gpu_memory_size = 0_u64
            Log.debug { "CudaMatrix.finalize: Freed GPU memory. Remaining: #{@@active_matrices} matrices, #{@@total_gpu_memory_allocated} bytes" }
          rescue ex
            Log.warn { "CudaMatrix.finalize: Failed to free GPU memory for #{@rows}x#{@cols}: #{ex}" }
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
      m.sync_to_device! if CUDA.fully_available?
      m
    end

    # Return the transposed matrix. When CUDA is available the returned
    # instance keeps the device buffer in sync so that further GPU
    # operations can be used without additional copies.
    def transpose
      result = CudaMatrix.new(@cols, @rows)

      # Use GPU kernel for transpose when CUDA is fully available
      if CUDA.fully_available? && (src_ptr = self.device_ptr) && (dst_ptr = result.device_ptr) &&
         !src_ptr.null? && !dst_ptr.null?
        begin
          # Make sure source data is on GPU
          self.sync_to_device! unless device_dirty?

          # Double-check pointers are still valid after sync
          src_ptr_check = self.device_ptr
          dst_ptr_check = result.device_ptr
          return transpose_cpu if !src_ptr_check || src_ptr_check.null? || !dst_ptr_check || dst_ptr_check.null?

          # Use GPU kernel for transpose with error handling
          CUDA.transpose(dst_ptr, src_ptr, @rows, @cols)

          # Mark result as dirty on device
          result.mark_device_dirty!
          return result
        rescue e
          # GPU operation failed, fall back to CPU
          Log.warn { "GPU transpose failed (#{e}), falling back to CPU" }
          return transpose_cpu
        end
      end

      # CPU fallback
      transpose_cpu
    end

    private def transpose_cpu
      result = CudaMatrix.new(@cols, @rows)

      if CUDA.fully_available? && device_dirty?
        # Keep the transpose operation minimal and let GPU handle it later
        # For now, sync to CPU and do CPU transpose, but mark result for GPU
        sync_from_device!
      end

      @rows.times do |i|
        @cols.times do |j|
          result.unsafe_set(j, i, unsafe_get(i, j))
        end
      end
      result.sync_to_device! if CUDA.fully_available?
      result
    end

    def sync_to_device!
      return unless dptr = @device_ptr
      return if dptr.null?

      Log.debug { "CudaMatrix.sync_to_device!: Syncing #{@rows}x#{@cols} matrix to GPU at #{dptr.address}" }

      begin
        # Use regular memory for host-device transfer (avoiding pinned memory limits)
        size = @rows * @cols
        bytes = (size * 8).to_u64

        # Create a stable buffer that won't be moved by GC
        buffer = Slice(Float64).new(size) do |i|
          row = i // @cols
          col = i % @cols
          unsafe_get(row, col)
        end

        copy_result = CUDA.memcpy(dptr.as(Pointer(Void)), buffer.to_unsafe.as(Pointer(Void)), bytes, CUDA::MemcpyKind::HostToDevice)

        if copy_result != 0
          Log.error { "CudaMatrix.sync_to_device!: GPU memcpy failed with result #{copy_result} for #{@rows}x#{@cols}" }
          @device_ptr = Pointer(Float64).null
        else
          Log.debug { "CudaMatrix.sync_to_device!: Successfully synced #{@rows}x#{@cols} to GPU" }
          mark_device_clean!
        end
      rescue ex : Exception
        Log.error { "CudaMatrix.sync_to_device!: Exception during sync for #{@rows}x#{@cols}: #{ex}" }
        @device_ptr = Pointer(Float64).null
      end
    end

    def sync_from_device!
      return unless dptr = @device_ptr
      return if dptr.null?
      return unless device_dirty? # Only sync if GPU data is newer

      begin
        size = @rows * @cols
        bytes = (size * 8).to_u64

        # Use regular memory copy (avoiding pinned memory limits)
        buffer = Slice(Float64).new(size, 0.0)
        copy_result = CUDA.memcpy(buffer.to_unsafe.as(Pointer(Void)), dptr.as(Pointer(Void)), bytes, CUDA::MemcpyKind::DeviceToHost)

        if copy_result == 0
          size.times do |i|
            row = i // @cols
            col = i % @cols
            unsafe_set(row, col, buffer[i])
          end
          mark_device_clean!
        else
          @device_ptr = Pointer(Float64).null
        end
      rescue
        @device_ptr = Pointer(Float64).null
      end
    end

    def slice_cols(start_col : Int32, length : Int32)
      result = CudaMatrix.new(@rows, length)
      if CUDA.fully_available? && (sptr = self.device_ptr) && (dptr = result.device_ptr) && !sptr.null? && !dptr.null?
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
      result.sync_to_device! if CUDA.fully_available?
      result
    end

    def set_cols!(start_col : Int32, other : CudaMatrix)
      raise ArgumentError.new("row mismatch") unless other.rows == @rows
      if CUDA.fully_available? && (dptr = self.device_ptr) && (sptr = other.device_ptr) && !dptr.null? && !sptr.null?
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
      self.sync_to_device! if CUDA.fully_available?
      self
    end

    def *(other : CudaMatrix)
      raise ArgumentError.new("size mismatch for multiplication") unless @cols == other.rows

      Log.debug { "CudaMatrix.*: Multiplying #{@rows}x#{@cols} * #{other.rows}x#{other.cols}" }

      if CUDA.fully_available? && (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?
        Log.debug { "CudaMatrix.*: Using GPU path for matrix multiplication" }
        # Ensure both operands have up-to-date GPU data
        self.sync_to_device! unless device_dirty?
        other.sync_to_device! unless other.device_dirty?

        handle = CUDA.create_handle
        result = CudaMatrix.new(@rows, other.cols)
        # CUBLAS assumes column-major, but we use row-major
        # To compute C = A * B in row-major, we compute C^T = B^T * A^T
        # So we swap the order: gemm(B, A, C) with dimensions swapped
        CUDA.gemm(handle, ptr_b, ptr_a, result.device_ptr.not_nil!,
          other.cols, @rows, other.rows,
          other.cols, @cols, result.cols)
        CUDA.destroy_handle(handle)

        # Mark result as having newer GPU data
        result.mark_device_dirty!
        Log.debug { "CudaMatrix.*: GPU matrix multiplication completed successfully" }
        result
      else
        # CPU fallback
        Log.warn { "CudaMatrix.*: Falling back to CPU for matrix multiplication - CUDA.fully_available?=#{CUDA.fully_available?}, self.device_ptr=#{device_ptr ? "valid" : "null"}, other.device_ptr=#{other.device_ptr ? "valid" : "null"}" }
        raise ArgumentError.new("size mismatch for multiplication") unless @cols == other.rows
        result = CudaMatrix.new(@rows, other.cols)
        @rows.times do |i|
          other.cols.times do |j|
            sum = 0.0
            @cols.times do |k|
              sum += self[i, k] * other[k, j]
            end
            result.unsafe_set(i, j, sum)
          end
        end
        result.sync_to_device! if CUDA.fully_available?
        result
      end
    end

    # Clean CudaMatrix + CudaMatrix addition
    def +(other : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      if CUDA.fully_available? && (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?
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
        result.sync_to_device! if CUDA.fully_available?
        result
      end
    end

    def clone
      dup = CudaMatrix.new(@rows, @cols)
      if CUDA.fully_available? && (sptr = self.device_ptr) && (dptr = dup.device_ptr) && !sptr.null? && !dptr.null?
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
      dup.sync_to_device! if CUDA.fully_available?
      dup
    end

    # In-place element-wise addition.
    def add!(other : CudaMatrix)
      raise ArgumentError.new("size mismatch") unless other.rows == @rows && other.cols == @cols
      if CUDA.fully_available? && (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?
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
        self.sync_to_device! if CUDA.fully_available?
      end
      self
    end

    def -(other : CudaMatrix)
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      if CUDA.fully_available? && (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?
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
        result.sync_to_device! if CUDA.fully_available?
        result
      end
    end

    def *(scalar : Number)
      if CUDA.fully_available? && (dptr = self.device_ptr) && !dptr.null?
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
        result.sync_to_device! if CUDA.fully_available?
        result
      end
    end

    # Add a bias row vector to each row in-place.
    def add_bias!(bias : CudaMatrix)
      raise ArgumentError.new("bias size mismatch") unless bias.rows == 1 && bias.cols == @cols
      if CUDA.fully_available? && (dptr = self.device_ptr) && (bptr = bias.device_ptr) && !dptr.null? && !bptr.null?
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
        self.sync_to_device! if CUDA.fully_available?
      end
      self
    end

    # Element-wise ReLU activation in-place.
    def relu!
      if CUDA.fully_available? && (dptr = self.device_ptr) && !dptr.null?
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
        self.sync_to_device! if CUDA.fully_available?
      end
      self
    end

    # Multiply each column by the corresponding value in a row vector in-place.
    def mul_row_vector!(vec : CudaMatrix)
      raise ArgumentError.new("vector size mismatch") unless vec.rows == 1 && vec.cols == @cols

      if CUDA.fully_available? && (dptr = self.device_ptr) && (vptr = vec.device_ptr) && !dptr.null? && !vptr.null?
        # Use GPU kernel for column-wise scaling
        CUDA.mul_row_vector(dptr, vptr, @rows, @cols)
        # Mark result as dirty on device
        mark_device_dirty!
      else
        # CPU fallback
        @rows.times do |i|
          @cols.times do |j|
            self[i, j] *= vec[0, j]
          end
        end
        self.sync_to_device! if CUDA.fully_available?
      end
      self
    end

    # Convert CudaMatrix to SimpleMatrix for CPU operations
    def to_simple : SimpleMatrix
      sync_from_device! if device_dirty?
      result = SimpleMatrix.new(@rows, @cols)
      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = @data[i * @cols + j]
        end
      end
      result
    end

    # Helper methods for device_dirty flag
    def device_dirty?
      @device_dirty
    end

    def mark_device_dirty!
      @device_dirty = true
    end

    def mark_device_clean!
      @device_dirty = false
    end

    def to_a
      Array.new(@rows) do |i|
        Array.new(@cols) do |j|
          self[i, j]
        end
      end
    end

    # Force cleanup of GPU memory for this matrix
    def cleanup!
      if dptr = @device_ptr
        unless dptr.null?
          Log.debug { "CudaMatrix.cleanup!: Explicitly freeing #{@gpu_memory_size} bytes GPU memory at #{dptr.address} for #{@rows}x#{@cols}" }
          CUDA.free(dptr.as(Pointer(Void)))
          @@total_gpu_memory_allocated -= @gpu_memory_size
          @@active_matrices -= 1
          @device_ptr = Pointer(Float64).null
          @gpu_memory_size = 0_u64
        end
      end
    end
  end
end
