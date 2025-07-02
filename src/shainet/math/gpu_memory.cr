require "../cuda"

module SHAInet
  # GPU Memory Manager - helps minimize CPU-GPU transfers
  module GPUMemory
    extend self

    # -- Simple GPU allocator -------------------------------------------------
    @@pool = Hash(Int32, Array(Pointer(Float64))).new { |h, k| h[k] = [] of Pointer(Float64) }
    @@pool_limit : Int32 = 2 # Very small pool to reduce memory pressure

    # Configure the maximum number of cached buffers
    def pool_limit
      @@pool_limit
    end

    def pool_limit=(limit : Int32)
      @@pool_limit = limit
    end

    # Preallocate +count+ buffers of given shape
    def preallocate!(rows : Int32, cols : Int32, count : Int32)
      return unless CUDA.fully_available?
      size = rows * cols
      count.times do
        ptr = Pointer(Float64).null
        bytes = ((size) * 8).to_u64
        res = CUDA.malloc(pointerof(ptr).as(Pointer(Pointer(Void))), bytes)
        next unless res == 0
        @@pool[size] << ptr
      end
    end

    # Allocate device memory, reusing cached buffers when possible
    def alloc_buffer(rows : Int32, cols : Int32)
      return Pointer(Float64).null unless CUDA.fully_available?

      size = rows * cols

      # Sanity check - prevent excessive memory allocation
      if size <= 0 || size > 100_000_000 # 100M elements = ~800MB
        return Pointer(Float64).null
      end

      bucket = @@pool[size]?
      if bucket && !bucket.empty?
        ptr = bucket.pop
        return ptr unless ptr.null?
      end

      begin
        ptr = Pointer(Float64).null
        bytes = ((size) * 8).to_u64
        res = CUDA.malloc(pointerof(ptr).as(Pointer(Pointer(Void))), bytes)
        if res == 0 && !ptr.null?
          ptr
        else
          Pointer(Float64).null
        end
      rescue
        Pointer(Float64).null
      end
    end

    # Return a buffer to the pool for reuse
    def release_buffer(ptr : Pointer(Float64), rows : Int32, cols : Int32)
      return if ptr.null?
      size = rows * cols
      bucket = @@pool[size]?
      if bucket.nil?
        bucket = [] of Pointer(Float64)
        @@pool[size] = bucket
      end
      arr = bucket.not_nil!
      if arr.size < @@pool_limit
        arr << ptr
      else
        CUDA.free(ptr.as(Pointer(Void)))
      end
    end

    # Free all cached buffers
    def cleanup
      @@pool.each_value do |arr|
        arr.each do |ptr|
          CUDA.free(ptr.as(Pointer(Void)))
        end
      end
      @@pool.clear
    end

    # Convert SimpleMatrix to CudaMatrix if CUDA is available and input is not already CudaMatrix
    def to_gpu(matrix : SimpleMatrix)
      return matrix if matrix.is_a?(CudaMatrix) || !CUDA.fully_available?

      result = CudaMatrix.new(matrix.rows, matrix.cols)
      matrix.rows.times do |i|
        matrix.cols.times do |j|
          result[i, j] = matrix[i, j]
        end
      end
      result.sync_to_device!
      result
    end

    # Ensure matrix stays on GPU if it's already there
    def keep_on_gpu(matrix : SimpleMatrix)
      if matrix.is_a?(CudaMatrix)
        matrix
      elsif CUDA.fully_available?
        to_gpu(matrix)
      else
        matrix
      end
    end

    # Create a new matrix of the same type as the input
    def like(matrix : SimpleMatrix, rows : Int32, cols : Int32, init : Float64 = 0.0)
      if matrix.is_a?(CudaMatrix) && CUDA.fully_available?
        result = CudaMatrix.new(rows, cols, init)
        result.sync_to_device!
        result
      else
        SimpleMatrix.new(rows, cols, init)
      end
    end

    # Create zeros matrix of same type as input
    def zeros_like(matrix : SimpleMatrix, rows : Int32, cols : Int32)
      like(matrix, rows, cols, 0.0)
    end

    # Create ones matrix of same type as input
    def ones_like(matrix : SimpleMatrix, rows : Int32, cols : Int32)
      like(matrix, rows, cols, 1.0)
    end

    # Apply operation and ensure result stays on same device
    def preserve_device(input : SimpleMatrix, &block : SimpleMatrix -> SimpleMatrix)
      result = yield input

      # Ensure result is same type as input
      if input.is_a?(CudaMatrix) && !result.is_a?(CudaMatrix) && CUDA.fully_available?
        gpu_result = to_gpu(result)
        gpu_result
      else
        result
      end
    end

    # Batch sync multiple CudaMatrix objects from device efficiently
    def batch_sync_from_device(matrices : Array(SimpleMatrix))
      matrices.each do |matrix|
        if matrix.is_a?(CudaMatrix)
          matrix.sync_from_device!
        end
      end
    end

    # Batch sync multiple CudaMatrix objects to device efficiently
    def batch_sync_to_device(matrices : Array(SimpleMatrix))
      matrices.each do |matrix|
        if matrix.is_a?(CudaMatrix)
          matrix.sync_to_device!
        end
      end
    end

    # Check if all matrices in array are of the same type (all GPU or all CPU)
    def same_device_type?(matrices : Array(SimpleMatrix))
      return true if matrices.empty?

      first_type = matrices.first.class
      matrices.all? { |m| m.class == first_type }
    end

    # Convert all matrices to the same device type (prefer GPU if available)
    def unify_device_type(matrices : Array(SimpleMatrix))
      return matrices if same_device_type?(matrices)

      # If CUDA is available and any matrix is on GPU, move all to GPU
      has_gpu = matrices.any? { |m| m.is_a?(CudaMatrix) }

      if has_gpu && CUDA.fully_available?
        matrices.map { |m| to_gpu(m) }
      else
        # Otherwise ensure all are CPU matrices (SimpleMatrix)
        matrices.map do |m|
          if m.is_a?(CudaMatrix)
            m.sync_from_device!
            cpu_matrix = SimpleMatrix.new(m.rows, m.cols)
            m.rows.times do |i|
              m.cols.times do |j|
                cpu_matrix[i, j] = m[i, j]
              end
            end
            cpu_matrix
          else
            m
          end
        end
      end
    end

    # Memory usage helpers
    def gpu_memory_allocated?(matrix : SimpleMatrix)
      matrix.is_a?(CudaMatrix) && matrix.device_ptr && !matrix.device_ptr.not_nil!.null?
    end

    def estimate_gpu_memory_usage(matrices : Array(SimpleMatrix))
      total_elements = 0_i64
      matrices.each do |matrix|
        if gpu_memory_allocated?(matrix)
          total_elements += matrix.rows.to_i64 * matrix.cols.to_i64
        end
      end
      total_elements * 8 # 8 bytes per Float64
    end
  end
end
