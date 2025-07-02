require "./cuda_matrix"

module SHAInet
  class CudaMatrix
    def softmax_rows
      result = CudaMatrix.new(@rows, @cols)
      if CUDA.available? && (dptr = self.device_ptr) && !dptr.null? && (rptr = result.device_ptr) && !rptr.null?
        begin
          # Ensure source has up-to-date GPU data
          self.sync_to_device! unless device_dirty?

          CUDA.softmax_rows(rptr, dptr, @rows, @cols)

          # Mark result as having newer GPU data
          result.mark_device_dirty!
          return result
        rescue
        end
      end
      @rows.times do |i|
        sum = 0.0
        @cols.times { |j| sum += Math.exp(self[i, j]) }
        @cols.times { |j| result[i, j] = Math.exp(self[i, j]) / sum }
      end
      result.sync_to_device! if CUDA.available?
      result
    end

    def dropout(drop_percent : Int32)
      raise ArgumentError.new("drop_percent must be between 0 and 100") unless 0 <= drop_percent <= 100
      result = CudaMatrix.new(@rows, @cols)
      prob = drop_percent.to_f / 100.0
      if CUDA.available? && (dptr = self.device_ptr) && !dptr.null? && (rptr = result.device_ptr) && !rptr.null?
        seed = Random.rand(UInt64)
        begin
          # Ensure source has up-to-date GPU data
          self.sync_to_device! unless device_dirty?

          CUDA.dropout(rptr, dptr, @rows, @cols, prob, seed)

          # Mark result as having newer GPU data
          result.mark_device_dirty!
          return result
        rescue
        end
      end
      @rows.times do |i|
        @cols.times do |j|
          result[i, j] = rand < prob ? 0.0 : self[i, j]
        end
      end
      result.sync_to_device! if CUDA.available?
      result
    end
  end
end
