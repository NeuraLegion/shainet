require "./simple_matrix"
{% if flag?(:enable_cuda) %}
  require "../cuda"
  require "../cudnn"
{% else %}
  require "../cuda_stub"
{% end %}

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

    # Sync counters for performance tracking
    @@sync_to_device_count = 0_u64
    @@sync_from_device_count = 0_u64
    @@total_sync_bytes_to_device = 0_u64
    @@total_sync_bytes_from_device = 0_u64

    # Matrix creation tracking
    @@matrix_creation_count = 0_u64

    # Track allocation sites (callers)
    @@allocation_sites = Hash(String, UInt64).new(0_u64)

    # Detailed sync tracking by source
    @@sync_sources = Hash(String, UInt64).new(0_u64)

    # Disable workspace pool - use in-place operations instead
    @@matrix_pool = Hash(String, Array(CudaMatrix)).new { |h, k| h[k] = [] of CudaMatrix }
    @@pool_enabled = true
    @@max_pool_size = 5000

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

    def self.sync_stats
      {
        sync_to_device_count:         @@sync_to_device_count,
        sync_from_device_count:       @@sync_from_device_count,
        total_sync_bytes_to_device:   @@total_sync_bytes_to_device,
        total_sync_bytes_from_device: @@total_sync_bytes_from_device,
        matrix_creation_count:        @@matrix_creation_count,
      }
    end

    def self.reset_sync_stats
      @@sync_to_device_count = 0_u64
      @@sync_from_device_count = 0_u64
      @@total_sync_bytes_to_device = 0_u64
      @@total_sync_bytes_from_device = 0_u64
      @@matrix_creation_count = 0_u64
      @@sync_sources.clear
      @@allocation_sites.clear
    end

    def self.print_detailed_stats
      Log.debug { "GPU Memory Statistics:" }
      Log.debug { "  Total attempts: #{@@allocation_attempts}" }
      Log.debug { "  Failed attempts: #{@@allocation_failures}" }
      Log.debug { "  Success rate: #{@@allocation_attempts > 0 ? (100.0 * (@@allocation_attempts - @@allocation_failures) / @@allocation_attempts).round(2) : 0}%" }
      Log.debug { "  Active matrices: #{@@active_matrices}" }
      Log.debug { "  Total GPU memory: #{@@total_gpu_memory_allocated} bytes (#{(@@total_gpu_memory_allocated / 1024.0 / 1024.0).round(2)} MB)" }
      Log.debug { "  Memory limit: #{@@max_gpu_memory} bytes (#{(@@max_gpu_memory / 1024.0 / 1024.0).round(2)} MB)" }
      Log.debug { "  Usage %: #{(100.0 * @@total_gpu_memory_allocated / @@max_gpu_memory).round(2)}%" }
      Log.debug { "  Average size per matrix: #{@@active_matrices > 0 ? (@@total_gpu_memory_allocated / @@active_matrices).round(2) : 0} bytes" }
      Log.debug { "Allocation sites (top 20): #{SHAInet::CudaMatrix.print_top_allocation_sites(20)} " }
    end

    def self.print_top_allocation_sites(limit = 20)
      Log.debug { "Top CudaMatrix allocation sites:" }
      @@allocation_sites.to_a.sort_by { |(_, v)| v }.reverse.first(limit).each do |site, count|
        Log.debug { "%6d  %s" % {count, site} }
      end
    end

    def self.reset_allocation_sites
      @@allocation_sites.clear
    end

    def initialize(@rows : Int32, @cols : Int32, init : Float64 = 0.0)
      @data = Array(Float64).new(@rows * @cols, init)
      @device_ptr = Pointer(Float64).null

      # Count matrix creation
      @@matrix_creation_count += 1

      # Track allocation site (top non-cuda_matrix.cr frame)
      if call = caller.find { |c| !c.includes?("cuda_matrix.cr") }
        @@allocation_sites[call] += 1
      end

      # CudaMatrix requires CUDA to be available
      raise RuntimeError.new("CudaMatrix requires CUDA to be available") unless CUDA.fully_available?
      # Print the most frequent allocation sites
      size = @rows * @cols
      bytes = (size * 8).to_u64

      # Check if we would exceed memory limits or are getting close
      if @@total_gpu_memory_allocated + bytes > @@max_gpu_memory ||
         @@total_gpu_memory_allocated > (@@max_gpu_memory * 0.8).to_u64 # 80% threshold
        Log.warn { "CudaMatrix.initialize: GPU memory usage high (#{@@total_gpu_memory_allocated}/#{@@max_gpu_memory} bytes, #{@@active_matrices} matrices). Forcing cleanup..." }

        # Try again after cleanup
        if @@total_gpu_memory_allocated + bytes > @@max_gpu_memory
          raise RuntimeError.new("GPU memory limit exceeded: would use #{@@total_gpu_memory_allocated + bytes}/#{@@max_gpu_memory} bytes")
        end
      end

      @@allocation_attempts += 1

      ptr = Pointer(Float64).null
      result = CUDA.malloc(pointerof(ptr).as(Pointer(Pointer(Void))), bytes)

      if result == 0 && !ptr.null?
        @device_ptr = ptr
        @gpu_memory_size = bytes
        @@total_gpu_memory_allocated += bytes
        @@active_matrices += 1
      else
        @@allocation_failures += 1
        Log.error { "CudaMatrix.initialize: GPU allocation failed with result #{result} for #{@rows}x#{@cols}. Total usage: #{@@active_matrices} matrices, #{@@total_gpu_memory_allocated} bytes" }
        raise RuntimeError.new("Failed to allocate #{bytes} bytes of GPU memory (CUDA error: #{result})")
      end
    end

    # Basic matrix access operations
    def [](row : Int32, col : Int32)
      # If GPU data is newer, sync it to CPU first
      sync_from_device!("element_access") if device_dirty?
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
      m.sync_to_device!("matrix_from_array")
      m
    end

    def self.zeros(rows : Int32, cols : Int32)
      # Create new matrix directly - zeros are often used for weight matrices that persist
      m = new(rows, cols)
      m.zero! # Use optimized GPU zero kernel
      m
    end

    def self.ones(rows : Int32, cols : Int32)
      # Create new matrix directly - ones are often used for weight matrices that persist
      m = new(rows, cols)
      m.fill!(1.0)
      m
    end

    def random_fill!(min : Float64 = -0.1, max : Float64 = 0.1)
      @rows.times do |i|
        @cols.times do |j|
          self[i, j] = Random.rand(min..max)
        end
      end
      sync_to_device!("random_fill")
      self
    end

    def finalize
      # Each CudaMatrix cleans up its own GPU memory directly
      if dptr = @device_ptr
        unless dptr.null?
          begin
            CUDA.free(dptr.as(Pointer(Void)))
            @@total_gpu_memory_allocated -= @gpu_memory_size
            @@active_matrices -= 1
            @device_ptr = Pointer(Float64).null
            @gpu_memory_size = 0_u64
          rescue ex
            Log.warn { "CudaMatrix.finalize: Failed to free GPU memory for #{@rows}x#{@cols}: #{ex}" }
          end
        end
      end
    end

    # Return the transposed matrix - CREATE NEW MATRIX (used sparingly)
    def transpose
      # Create new matrix directly - transpose is unavoidable allocation
      result = CudaMatrix.new(@cols, @rows)

      # Use GPU kernel for transpose - fail fast if not available
      raise RuntimeError.new("GPU transpose requires valid device pointers") unless (src_ptr = self.device_ptr) && (dst_ptr = result.device_ptr) && !src_ptr.null? && !dst_ptr.null?

      # Make sure source data is on GPU
      self.sync_to_device!("transpose_operation") unless device_dirty?

      # Use GPU kernel for transpose
      CUDA.transpose(dst_ptr, src_ptr, @rows, @cols)

      # Mark result as dirty on device
      result.mark_device_dirty!
      result
    end

    # Transpose the matrix into the provided destination matrix in-place.
    # Avoids allocating a new matrix when a persistent transpose is needed.
    def transpose_into!(dest : CudaMatrix)
      raise ArgumentError.new("size mismatch") unless dest.rows == @cols && dest.cols == @rows
      raise RuntimeError.new("GPU transpose requires valid device pointers") unless (src_ptr = self.device_ptr) && (dst_ptr = dest.device_ptr) && !src_ptr.null? && !dst_ptr.null?
      # Ensure source data is on the GPU
      self.sync_to_device!("transpose_into") unless device_dirty?

      # Perform transpose using CUDA kernel
      CUDA.transpose(dst_ptr, src_ptr, @rows, @cols)
      dest.mark_device_dirty!
      dest
    end

    def self.track_sync(source : String)
      @@sync_sources[source] += 1
    end

    def self.sync_sources_stats
      @@sync_sources.to_h
    end

    def self.reset_sync_sources
      @@sync_sources.clear
    end

    def sync_to_device!(source : String = "unknown")
      return unless dptr = @device_ptr
      return if dptr.null?

      begin
        # Use regular memory for host-device transfer (avoiding pinned memory limits)
        size = @rows * @cols
        bytes = (size * 8).to_u64

        # Track sync operations for performance monitoring
        @@sync_to_device_count += 1
        @@total_sync_bytes_to_device += bytes
        self.class.track_sync("to_device:#{source}")

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
          mark_device_clean!
        end
      rescue ex : Exception
        Log.error { "CudaMatrix.sync_to_device!: Exception during sync for #{@rows}x#{@cols}: #{ex}" }
        @device_ptr = Pointer(Float64).null
      end
    end

    def sync_from_device!(source : String = "unknown")
      return unless dptr = @device_ptr
      return if dptr.null?
      return unless device_dirty? # Only sync if GPU data is newer

      begin
        size = @rows * @cols
        bytes = (size * 8).to_u64

        # Track sync operations for performance monitoring
        @@sync_from_device_count += 1
        @@total_sync_bytes_from_device += bytes
        self.class.track_sync("from_device:#{source}")

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

    # Slice a range of columns into an existing destination matrix using the
    # CUDA `slice_cols` kernel.
    def slice_cols_into!(dest : CudaMatrix, start_col : Int32, length : Int32)
      raise ArgumentError.new("size mismatch") unless dest.rows == @rows && dest.cols == length
      raise RuntimeError.new("GPU slice_cols_into! requires valid device pointers") unless (sptr = self.device_ptr) && (dptr = dest.device_ptr) && !sptr.null? && !dptr.null?

      # Ensure source data is on the GPU
      self.sync_to_device!("slice_cols_into") unless device_dirty?

      CUDA.slice_cols(dptr, sptr, @rows, @cols, start_col, length)

      dest.mark_device_dirty!
      dest
    end

    def slice_cols(start_col : Int32, length : Int32)
      result = CudaMatrix.new(@rows, length)
      slice_cols_into!(result, start_col, length)
      result
    end

    def set_cols!(start_col : Int32, other : CudaMatrix)
      raise ArgumentError.new("row mismatch") unless other.rows == @rows
      raise RuntimeError.new("GPU set_cols! requires valid device pointers") unless (dptr = self.device_ptr) && (sptr = other.device_ptr) && !dptr.null? && !sptr.null?

      # Ensure both matrices have up-to-date GPU data
      self.sync_to_device!("set_cols") unless device_dirty?
      other.sync_to_device!("set_cols") unless other.device_dirty?

      CUDA.set_cols(dptr, sptr, @rows, @cols, start_col, other.cols)

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # Set a specific row from another matrix's row
    def set_row!(row_idx : Int32, other : CudaMatrix, source_row : Int32 = 0)
      raise ArgumentError.new("column mismatch") unless other.cols == @cols
      raise ArgumentError.new("row index out of bounds") unless row_idx >= 0 && row_idx < @rows
      raise ArgumentError.new("source row index out of bounds") unless source_row >= 0 && source_row < other.rows

      # For now, use GPU copy operations to copy the row
      # This is more efficient than element-by-element access
      self.sync_to_device!("set_row") unless device_dirty?
      other.sync_to_device!("set_row") unless other.device_dirty?

      dptr = self.device_ptr
      sptr = other.device_ptr
      raise RuntimeError.new("GPU set_row! requires valid device pointers") unless dptr && sptr && !dptr.null? && !sptr.null?

      # Calculate pointers to the specific rows
      dest_row_ptr = dptr + (row_idx * @cols)
      src_row_ptr = sptr + (source_row * other.cols)

      # Copy the row data (cols * 8 bytes for Float64)
      bytes = (@cols * 8).to_u64
      CUDA.copy_device_to_device(dest_row_ptr, src_row_ptr, bytes)

      mark_device_dirty!
      self
    end

    # Optimized cuBLAS matrix multiplication
    def *(other : CudaMatrix)
      raise ArgumentError.new("size mismatch for multiplication") unless @cols == other.rows
      raise RuntimeError.new("GPU multiplication requires valid device pointers") unless (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?

      # Ensure both operands have up-to-date GPU data
      self.sync_to_device!("matrix_multiply") unless device_dirty?
      other.sync_to_device!("matrix_multiply") unless other.device_dirty?

      # Create result matrix directly - matrix multiplication creates new data
      result = CudaMatrix.new(@rows, other.cols)
      raise RuntimeError.new("Failed to allocate result matrix on GPU") unless result.device_ptr && !result.device_ptr.not_nil!.null?

      handle = CUDA.create_handle
      begin
        # Optimized cuBLAS GEMM - account for row-major vs column-major difference
        # To compute C = A * B in row-major, we compute C^T = B^T * A^T
        # So we swap the order: gemm(B, A, C) with dimensions swapped
        CUDA.gemm(handle, ptr_b, ptr_a, result.device_ptr.not_nil!,
          other.cols, @rows, other.rows,
          other.cols, @cols, result.cols)
      ensure
        CUDA.destroy_handle(handle)
      end

      # Mark result as having newer GPU data
      result.mark_device_dirty!
      result
    end

    # Clean CudaMatrix + CudaMatrix addition - optimized with cuDNN and cuBLAS
    def +(other : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      raise RuntimeError.new("GPU addition requires valid device pointers") unless (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?

      # Ensure both operands have up-to-date GPU data
      self.sync_to_device!("matrix_addition") unless device_dirty?
      other.sync_to_device!("matrix_addition") unless other.device_dirty?

      # Create result matrix directly - don't use workspace pool for arithmetic operations
      result = CudaMatrix.new(@rows, @cols)
      raise RuntimeError.new("Failed to allocate result matrix on GPU") unless result.device_ptr && !result.device_ptr.not_nil!.null?

      # Try cuDNN first for element-wise operations
      if CUDNN.available?
        begin
          CUDNN.element_add!(result, self, other, 1.0, 1.0)
          return result
        rescue e : Exception
          Log.error { "cuDNN element_add failed: #{e}, falling back to cuBLAS" }
        end
      end

      # Fallback to cuBLAS GEAM
      handle = CUDA.create_handle
      begin
        CUDA.geam(handle, ptr_a, ptr_b, result.device_ptr.not_nil!, @rows, @cols, 1.0, 1.0)
      ensure
        CUDA.destroy_handle(handle)
      end

      result.mark_device_dirty!
      result
    end

    # Clean CudaMatrix - CudaMatrix subtraction - optimized with cuBLAS and workspace pool
    def -(other : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      raise RuntimeError.new("GPU subtraction requires valid device pointers") unless (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?

      # Ensure both operands have up-to-date GPU data
      self.sync_to_device!("matrix_subtraction") unless device_dirty?
      other.sync_to_device!("matrix_subtraction") unless other.device_dirty?

      # Create result matrix directly - don't use workspace pool for arithmetic operations
      result = CudaMatrix.new(@rows, @cols)
      raise RuntimeError.new("Failed to allocate result matrix on GPU") unless result.device_ptr && !result.device_ptr.not_nil!.null?

      handle = CUDA.create_handle
      begin
        # Use GEAM with alpha=1.0, beta=-1.0 to compute A - B
        CUDA.geam(handle, ptr_a, ptr_b, result.device_ptr.not_nil!, @rows, @cols, 1.0, -1.0)
      ensure
        CUDA.destroy_handle(handle)
      end

      # Mark result as having newer GPU data
      result.mark_device_dirty!
      result
    end

    def clone
      dup = CudaMatrix.new(@rows, @cols)
      raise RuntimeError.new("GPU clone requires valid device pointers") unless (sptr = self.device_ptr) && (dptr = dup.device_ptr) && !sptr.null? && !dptr.null?

      # If we have GPU data, copy it directly on GPU
      if device_dirty?
        # GPU -> GPU copy
        bytes = (@rows * @cols * 8).to_u64
        result = CUDA.memcpy(dptr.as(Pointer(Void)), sptr.as(Pointer(Void)), bytes, CUDA::MemcpyKind::DeviceToDevice)
        raise RuntimeError.new("GPU-to-GPU memcpy failed") if result != 0

        dup.mark_device_dirty!
        return dup
      end

      # CPU -> GPU copy (sync to device)
      @rows.times do |i|
        @cols.times do |j|
          dup.unsafe_set(i, j, unsafe_get(i, j))
        end
      end
      dup.sync_to_device!("matrix_clone")
      dup
    end

    # In-place element-wise addition - optimized with cuDNN and cuBLAS.
    def add!(other : CudaMatrix)
      raise ArgumentError.new("size mismatch") unless other.rows == @rows && other.cols == @cols
      raise RuntimeError.new("GPU add! requires valid device pointers") unless (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?

      # Ensure both matrices have up-to-date GPU data
      self.sync_to_device!("matrix_add_inplace") unless device_dirty?
      other.sync_to_device!("matrix_add_inplace") unless other.device_dirty?

      # Try cuDNN first for element-wise operations
      if CUDNN.available?
        begin
          CUDNN.element_add!(self, self, other, 1.0, 1.0)
          return self
        rescue e : Exception
          Log.error { "cuDNN element_add failed: #{e}, falling back to cuBLAS" }
        end
      end

      # Fallback to cuBLAS GEAM
      handle = CUDA.create_handle
      begin
        CUDA.geam(handle, ptr_a, ptr_b, ptr_a, @rows, @cols, 1.0, 1.0)
      ensure
        CUDA.destroy_handle(handle)
      end

      mark_device_dirty!
      self
    end

    # In-place element-wise subtraction - optimized with cuBLAS.
    def sub!(other : CudaMatrix)
      raise ArgumentError.new("size mismatch") unless other.rows == @rows && other.cols == @cols
      raise RuntimeError.new("GPU sub! requires valid device pointers") unless (ptr_a = self.device_ptr) && (ptr_b = other.device_ptr) && !ptr_a.null? && !ptr_b.null?

      # Ensure both matrices have up-to-date GPU data
      self.sync_to_device!("matrix_sub_inplace") unless device_dirty?
      other.sync_to_device!("matrix_sub_inplace") unless other.device_dirty?

      handle = CUDA.create_handle
      begin
        CUDA.geam(handle, ptr_a, ptr_b, ptr_a, @rows, @cols, 1.0, -1.0)
      ensure
        CUDA.destroy_handle(handle)
      end

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # Fill matrix with a constant value in-place.
    def fill!(value : Float64)
      if CUDA.fully_available? && (dptr = device_ptr) && !dptr.null?
        # Special case for zero - use GPU kernel directly
        if value == 0.0
          size = @rows * @cols
          CUDA.zero_matrix(dptr, size)
          mark_device_dirty!
        else
          # For non-zero values, fall back to CPU approach
          # But only sync if actually needed
          sync_from_device!("matrix_fill") if device_dirty?

          @rows.times do |i|
            @cols.times do |j|
              unsafe_set(i, j, value)
            end
          end

          sync_to_device!("matrix_fill")
        end
      else
        # CPU fallback
        @rows.times do |i|
          @cols.times do |j|
            unsafe_set(i, j, value)
          end
        end
        mark_device_clean!
      end
      self
    end

    # Optimized scalar multiplication using cuBLAS SCAL
    def *(scalar : Number)
      raise RuntimeError.new("GPU scalar multiplication requires valid device pointer") unless (dptr = self.device_ptr) && !dptr.null?

      # Ensure self has up-to-date GPU data
      self.sync_to_device!("scalar_multiplication") unless device_dirty?

      # Create a copy to avoid modifying the original
      out = self.clone
      ptr = out.device_ptr.not_nil!

      handle = CUDA.create_handle
      begin
        CUDA.scal(handle, ptr, (@rows*@cols), scalar.to_f64)
      ensure
        CUDA.destroy_handle(handle)
      end

      # Mark result as having newer GPU data
      out.mark_device_dirty!
      out
    end

    # Add a bias row vector to each row in-place.
    def add_bias!(bias : CudaMatrix)
      raise ArgumentError.new("bias size mismatch") unless bias.rows == 1 && bias.cols == @cols

      # Use cuDNN for optimized bias addition if available
      if CUDNN.available?
        begin
          CUDNN.add_bias!(self, bias)
          return self
        rescue e : Exception
          Log.error { "cuDNN add_bias failed: #{e}, falling back to CUDA kernel" }
        end
      end

      # Fallback to CUDA kernel
      raise RuntimeError.new("GPU add_bias! requires valid device pointers") unless (dptr = self.device_ptr) && (bptr = bias.device_ptr) && !dptr.null? && !bptr.null?

      # Ensure both matrices have up-to-date GPU data
      self.sync_to_device!("bias_addition") unless device_dirty?
      bias.sync_to_device!("bias_addition") unless bias.device_dirty?

      CUDA.add_bias(dptr, bptr, @rows, @cols)

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # Element-wise ReLU activation in-place.
    def relu!
      # Use cuDNN for optimized ReLU if available
      if CUDNN.available?
        begin
          CUDNN.relu_forward(self, self)
          return self
        rescue e : Exception
          Log.error { "cuDNN ReLU failed: #{e}, falling back to CUDA kernel" }
        end
      end

      # Fallback to CUDA kernel
      raise RuntimeError.new("GPU ReLU requires valid device pointer") unless (dptr = self.device_ptr) && !dptr.null?

      # Ensure self has up-to-date GPU data
      self.sync_to_device!("relu_activation") unless device_dirty?

      CUDA.relu(dptr, (@rows*@cols))

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # Multiply each column by the corresponding value in a row vector in-place.
    def mul_row_vector!(vec : CudaMatrix)
      raise ArgumentError.new("vector size mismatch") unless vec.rows == 1 && vec.cols == @cols
      raise RuntimeError.new("GPU mul_row_vector! requires valid device pointers") unless (dptr = self.device_ptr) && (vptr = vec.device_ptr) && !dptr.null? && !vptr.null?

      # Ensure both matrices have up-to-date GPU data
      self.sync_to_device!("mul_row_vector") unless device_dirty?
      vec.sync_to_device!("mul_row_vector") unless vec.device_dirty?

      # Use GPU kernel for column-wise scaling
      CUDA.mul_row_vector(dptr, vptr, @rows, @cols)
      # Mark result as dirty on device
      mark_device_dirty!
      self
    end

    # Convert CudaMatrix to SimpleMatrix for CPU operations
    def to_simple : SimpleMatrix
      sync_from_device!("to_simple_conversion") if device_dirty?
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
      # Ensure CPU data is up to date (single sync instead of per-element)
      sync_from_device!("bulk_to_a") if device_dirty?

      # Use direct data array access to avoid repeated element access syncs
      Array.new(@rows) do |i|
        Array.new(@cols) do |j|
          @data[i * @cols + j]
        end
      end
    end

    # More efficient flat array conversion - avoids nested array creation
    def to_flat_array
      sync_from_device!("bulk_to_flat_array") if device_dirty?
      @data.dup
    end

    # Force cleanup of GPU memory for this matrix
    def cleanup!
      if dptr = @device_ptr
        unless dptr.null?
          CUDA.free(dptr.as(Pointer(Void)))
          @@total_gpu_memory_allocated -= @gpu_memory_size
          @@active_matrices -= 1
          @device_ptr = Pointer(Float64).null
          @gpu_memory_size = 0_u64
        end
      end
      sync_to_device!
      self
    end

    # Provide access to raw data for batch operations
    def raw_data
      @data
    end

    # Copy data from another CudaMatrix
    def copy_from!(other : CudaMatrix)
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      raise RuntimeError.new("GPU copy requires valid device pointers") unless (sptr = other.device_ptr) && (dptr = self.device_ptr) && !sptr.null? && !dptr.null?

      # Ensure source has up-to-date GPU data
      other.sync_to_device!("copy_from") unless other.device_dirty?

      # GPU -> GPU copy
      bytes = (@rows * @cols * 8).to_u64
      result = CUDA.memcpy(dptr.as(Pointer(Void)), sptr.as(Pointer(Void)), bytes, CUDA::MemcpyKind::DeviceToDevice)
      raise RuntimeError.new("GPU-to-GPU memcpy failed") if result != 0

      mark_device_dirty!
      self
    end

    # In-place zeroing using GPU kernel
    def zero!
      if CUDA.fully_available? && (dptr = device_ptr) && !dptr.null?
        size = @rows * @cols
        CUDA.zero_matrix(dptr, size)
        mark_device_dirty!
      else
        # CPU fallback - zero all elements
        @rows.times do |i|
          @cols.times do |j|
            unsafe_set(i, j, 0.0)
          end
        end
        # Mark CPU data as newer for CPU fallback
        mark_device_clean!
      end
      self
    end

    # Element-wise sigmoid activation in-place using cuDNN.
    def sigmoid!
      # Use cuDNN for optimized sigmoid
      if CUDNN.available?
        begin
          CUDNN.sigmoid_forward!(self, self)
          return self
        rescue e : Exception
          Log.error { "cuDNN sigmoid failed: #{e}, falling back to CUDA kernel" }
        end
      end

      # Fallback to CUDA kernel
      raise RuntimeError.new("GPU sigmoid requires valid device pointer") unless (dptr = self.device_ptr) && !dptr.null?

      # Ensure self has up-to-date GPU data
      self.sync_to_device!("sigmoid_activation") unless device_dirty?

      # Apply sigmoid in-place - use same pointer for all three parameters
      size = @rows * @cols
      CUDA.sigmoid_forward(dptr, dptr, dptr, size)

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # High-performance in-place scalar multiplication using cuBLAS SCAL
    def scale!(scalar : Float64)
      raise RuntimeError.new("GPU scale! requires valid device pointer") unless (dptr = self.device_ptr) && !dptr.null?

      # Ensure self has up-to-date GPU data
      self.sync_to_device!("scalar_scale_inplace") unless device_dirty?

      handle = CUDA.create_handle
      begin
        CUDA.scal(handle, dptr, (@rows*@cols), scalar)
      ensure
        CUDA.destroy_handle(handle)
      end

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # High-performance element-wise division
    def /(other : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols

      result = CudaMatrix.new(@rows, @cols)

      if CUDA.fully_available? && (sptr = self.device_ptr) && (optr = other.device_ptr) && (dptr = result.device_ptr) && !sptr.null? && !optr.null? && !dptr.null?
        # Ensure both operands have up-to-date GPU data
        self.sync_to_device!("element_division") unless device_dirty?
        other.sync_to_device!("element_division") unless other.device_dirty?

        size = @rows * @cols
        CUDA.element_div(dptr, sptr, optr, size)

        result.mark_device_dirty!
      else
        # Fallback to CPU implementation
        self.sync_from_device!("element_division") if device_dirty?
        other.sync_from_device!("element_division") if other.device_dirty?

        @rows.times do |i|
          @cols.times do |j|
            self_val = self.unsafe_get(i, j)
            other_val = other.unsafe_get(i, j)
            result.unsafe_set(i, j, other_val == 0.0 ? 0.0 : self_val / other_val)
          end
        end

        result.sync_to_device!("element_division_result")
      end

      result
    end

    # Element-wise softmax using cuDNN when available
    def softmax_rows!
      # Use cuDNN for optimized softmax if available
      if CUDNN.available?
        begin
          CUDNN.softmax_rows(self, self)
          return self
        rescue e : Exception
          Log.error { "cuDNN softmax failed: #{e}, falling back to CUDA kernel" }
        end
      end

      # Try custom CUDA kernel before falling back to CPU
      if CUDA.fully_available? && (dptr = self.device_ptr) && !dptr.null?
        begin
          self.sync_to_device!("softmax_kernel") unless device_dirty?
          CUDA.softmax_rows(dptr, dptr, @rows, @cols)
          mark_device_dirty!
          return self
        rescue e : Exception
          Log.error { "CUDA softmax kernel failed: #{e}, falling back to CPU" }
        end
      end

      # CPU fallback implementation
      raise RuntimeError.new("GPU softmax requires valid device pointer") unless (dptr = self.device_ptr) && !dptr.null?

      # Ensure self has up-to-date CPU data
      self.sync_from_device!("softmax_fallback")
      @rows.times do |i|
        # Compute softmax for each row
        row_max = -Float64::INFINITY
        @cols.times { |j| row_max = Math.max(row_max, unsafe_get(i, j)) }

        row_sum = 0.0
        @cols.times do |j|
          val = Math.exp(unsafe_get(i, j) - row_max)
          unsafe_set(i, j, val)
          row_sum += val
        end

        @cols.times { |j| unsafe_set(i, j, unsafe_get(i, j) / row_sum) }
      end
      self.sync_to_device!("softmax_result")

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # Get a matrix from the pool or create a new one
    def self.get_workspace(rows : Int32, cols : Int32, source : String = "workspace") : CudaMatrix
      return new(rows, cols) unless @@pool_enabled

      key = "#{rows}x#{cols}"
      pool = @@matrix_pool[key]

      if matrix = pool.pop?
        # Reuse existing matrix - zero it out for cleanliness
        matrix.zero!
        matrix
      else
        # Create new matrix
        new(rows, cols)
      end
    end

    # Return a matrix to the pool for reuse
    def self.return_workspace(matrix : CudaMatrix)
      return unless @@pool_enabled

      key = "#{matrix.rows}x#{matrix.cols}"
      pool = @@matrix_pool[key]

      # Only pool if we haven't exceeded the limit
      if pool.size < @@max_pool_size
        pool << matrix
      end
    end

    # Clear all pooled matrices
    def self.clear_workspace_pool
      total_freed = 0
      @@matrix_pool.each_value do |pool|
        total_freed += pool.size
        pool.clear
      end
    end

    # Get pool statistics
    def self.pool_stats
      total_pooled = @@matrix_pool.values.sum(&.size)
      {
        enabled:      @@pool_enabled,
        total_pooled: total_pooled,
        pools:        @@matrix_pool.transform_values(&.size),
      }
    end

    # In-place matrix multiplication with accumulation: self = alpha * A * B + beta * self
    def gemm!(a : CudaMatrix, b : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = 0.0)
      raise ArgumentError.new("size mismatch for in-place GEMM") unless a.cols == b.rows && @rows == a.rows && @cols == b.cols
      ptr_a = a.device_ptr
      ptr_b = b.device_ptr
      ptr_c = self.device_ptr
      if !ptr_a || !ptr_b || !ptr_c || ptr_a.null? || ptr_b.null? || ptr_c.null?
        raise RuntimeError.new("GPU in-place GEMM requires valid device pointers")
      end

      # Ensure all operands have up-to-date GPU data
      a.sync_to_device!("gemm_inplace") unless a.device_dirty?
      b.sync_to_device!("gemm_inplace") unless b.device_dirty?
      self.sync_to_device!("gemm_inplace") unless device_dirty?

      handle = CUDA.create_handle
      begin
        # In-place GEMM: C = alpha * A * B + beta * C
        # cuBLAS expects column-major ordering, so we perform the same
        # transpose trick used in `*` by swapping operands and dimensions.
        # Treating row-major A,B as column-major A^T,B^T results in:
        # C^T = B^T * A^T
        CUDA.gemm_accumulate(handle, ptr_b, ptr_a, ptr_c,
          b.cols, a.rows, b.rows,
          b.cols, a.cols, @cols, alpha, beta)
      ensure
        CUDA.destroy_handle(handle)
      end

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # In-place weight update: self = self - lr * gradient
    def weight_update!(gradient : CudaMatrix, learning_rate : Float64)
      raise ArgumentError.new("size mismatch for weight update") unless @rows == gradient.rows && @cols == gradient.cols
      raise RuntimeError.new("GPU weight update requires valid device pointers") unless (grad_ptr = gradient.device_ptr) && (weight_ptr = self.device_ptr) && !grad_ptr.null? && !weight_ptr.null?

      # Ensure both matrices have up-to-date GPU data
      self.sync_to_device!("weight_update") unless device_dirty?
      gradient.sync_to_device!("weight_update") unless gradient.device_dirty?

      handle = CUDA.create_handle
      begin
        # Use AXPY: weights = weights - lr * gradients
        total_elements = @rows * @cols
        CUDA.axpy(handle, -learning_rate, grad_ptr, weight_ptr, total_elements)
      ensure
        CUDA.destroy_handle(handle)
      end

      # Mark self as having newer GPU data
      mark_device_dirty!
      self
    end

    # Element-wise multiplication using cuDNN OpTensor
    def element_mul!(other : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = 0.0)
      # Use cuDNN for optimized element-wise multiplication
      if CUDNN.available?
        begin
          CUDNN.element_multiply!(self, self, other, alpha, beta)
          return self
        rescue e : Exception
          Log.error { "cuDNN element_mul failed: #{e}, falling back to CPU" }
        end
      end

      # CPU fallback
      raise ArgumentError.new("size mismatch") unless @rows == other.rows && @cols == other.cols
      self.sync_from_device!("cudnn_element_mul_fallback") if device_dirty?
      other.sync_from_device!("cudnn_element_mul_fallback") if other.device_dirty?

      @rows.times do |i|
        @cols.times do |j|
          self_val = self.unsafe_get(i, j)
          other_val = other.unsafe_get(i, j)
          result_val = alpha * self_val * other_val + beta * self_val
          self.unsafe_set(i, j, result_val)
        end
      end
      self.sync_to_device!("cudnn_element_mul_result")
      mark_device_dirty!
      self
    end

    # Dropout using custom CUDA kernel (always, since cuDNN does not support Float64)
    def dropout!(prob : Float64, seed : UInt64 = Random.rand(UInt64::MAX))
      if CUDA.fully_available? && (dptr = self.device_ptr) && !dptr.null?
        begin
          self.sync_to_device!("dropout_kernel") unless device_dirty?
          result = CUDA.dropout(dptr, (@rows * @cols), prob.to_f32, seed)
          if result == 0
            mark_device_dirty!
            return self
          end
        rescue e : Exception
          Log.error { "CUDA dropout kernel failed: #{e}, falling back to CPU" }
        end
      end

      # CPU fallback
      self.sync_from_device!("dropout_fallback") if device_dirty?
      @rows.times do |i|
        @cols.times do |j|
          result_val = Random.rand < prob ? 0.0 : self.unsafe_get(i, j)
          self.unsafe_set(i, j, result_val)
        end
      end
      self.sync_to_device!("dropout_result")
      mark_device_dirty!
      self
    end

    # Element-wise tanh activation using cuDNN
    def tanh!
      # Use cuDNN for optimized tanh
      if CUDNN.available?
        begin
          CUDNN.tanh_forward!(self, self)
          return self
        rescue e : Exception
          Log.error { "cuDNN tanh failed: #{e}, falling back to CPU" }
        end
      end

      # CPU fallback
      self.sync_from_device!("tanh_fallback") if device_dirty?
      @rows.times do |i|
        @cols.times do |j|
          val = self.unsafe_get(i, j)
          self.unsafe_set(i, j, Math.tanh(val))
        end
      end
      self.sync_to_device!("tanh_result")
      mark_device_dirty!
      self
    end
  end
end
