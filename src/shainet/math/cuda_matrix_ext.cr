require "./cuda_matrix"

module SHAInet
  class CudaMatrix
    def softmax_rows
      result = CudaMatrix.new(@rows, @cols)
      if CUDA.fully_available? && (dptr = self.device_ptr) && !dptr.null? && (rptr = result.device_ptr) && !rptr.null?
        begin
          # Ensure source has up-to-date GPU data
          self.sync_to_device! unless device_dirty?

          # Verify source data
          Log.debug { "Softmax source data verification" }
          test_buf = Array(Float64).new(@rows * @cols, 0.0)
          CUDA.memcpy(test_buf.to_unsafe.as(Pointer(Void)),
            dptr.as(Pointer(Void)),
            (@rows * @cols * 8).to_u64,
            CUDA::MemcpyKind::DeviceToHost)

          # Log a sample of the source data
          Log.debug { "Source data sample: #{test_buf[0...4].join(", ")}" }

          # Ensure result is zeroed
          zeroes = Array(Float64).new(@rows * @cols, 0.0)
          CUDA.memcpy(rptr.as(Pointer(Void)),
            zeroes.to_unsafe.as(Pointer(Void)),
            (@rows * @cols * 8).to_u64,
            CUDA::MemcpyKind::HostToDevice)

          # Run the kernel
          Log.debug { "Running CUDA softmax_rows with rows=#{@rows}, cols=#{@cols}" }
          CUDA.softmax_rows(rptr, dptr, @rows, @cols)

          # Check result data
          test_result = Array(Float64).new(@rows * @cols, 0.0)
          CUDA.memcpy(test_result.to_unsafe.as(Pointer(Void)),
            rptr.as(Pointer(Void)),
            (@rows * @cols * 8).to_u64,
            CUDA::MemcpyKind::DeviceToHost)

          # Log a sample of the result data
          Log.debug { "Result data sample: #{test_result[0...4].join(", ")}" }

          # Check if all results are zero
          if test_result.all? { |v| v == 0.0 }
            Log.error { "CUDA softmax_rows produced all zeros. Falling back to CPU." }
            raise "CUDA kernel failed silently"
          end

          # Mark result as having newer GPU data
          result.mark_device_dirty!
          return result
        rescue ex
          # Log the error and fall back to CPU
          Log.error { "CUDA softmax_rows failed: #{ex.message}. Falling back to CPU." }
        end
      end

      # CPU fallback - ensure we have current data from GPU
      self.sync_from_device! if device_dirty?

      @rows.times do |i|
        sum = 0.0
        @cols.times { |j| sum += Math.exp(self[i, j]) }
        @cols.times { |j| result[i, j] = Math.exp(self[i, j]) / sum }
      end
      result.sync_to_device! if CUDA.fully_available?
      result
    end

    def dropout(drop_percent : Int32)
      raise ArgumentError.new("drop_percent must be between 0 and 100") unless 0 <= drop_percent <= 100
      result = CudaMatrix.new(@rows, @cols)
      prob = drop_percent.to_f / 100.0
      if CUDA.fully_available? && (dptr = self.device_ptr) && !dptr.null? && (rptr = result.device_ptr) && !rptr.null?
        seed = Random.rand(UInt64)
        begin
          # Ensure source has up-to-date GPU data
          self.sync_to_device! unless device_dirty?

          CUDA.dropout(rptr, dptr, @rows, @cols, prob, seed)

          # Mark result as having newer GPU data
          result.mark_device_dirty!
          return result
        rescue ex
          # Log the error and fall back to CPU
          Log.error { "CUDA dropout failed: #{ex.message}. Falling back to CPU." }
        end
      end

      # CPU fallback - ensure we have current data from GPU
      self.sync_from_device! if device_dirty?

      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = rand < prob ? 0.0 : self[i, j]
        end
      end
      result.sync_to_device! if CUDA.fully_available?
      result
    end
  end
end
