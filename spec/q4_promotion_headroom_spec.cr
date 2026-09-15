require "./spec_helper"

# The promotion policy has to behave differently in two regimes, and a spec that only
# checked the easy one would pass while the guard did nothing:
#
#   * headroom      -> promote on FIRST touch, because there is no victim to protect
#   * at capacity   -> keep the frequency gate, so a cold one-off cannot displace a hot
#                      weight
#
# The second group is the guard failing in the other direction, and it is the reason the
# threshold is kept at all rather than deleted.
# The suite shares one process-global cache and budget, so every example bounds the budget
# and releases what it took. Leaking either wedges later examples.
def with_budget(bytes : UInt64, &)
  saved = SHAInet::Q4HostMatrix.budget_bytes
  SHAInet::Q4HostMatrix.release_cache!
  SHAInet::Q4HostMatrix.budget_bytes = bytes
  begin
    yield
  ensure
    SHAInet::Q4HostMatrix.release_cache!
    SHAInet::Q4HostMatrix.budget_bytes = saved
  end
end

def host_weight(k : Int32, n : Int32) : SHAInet::Q4HostMatrix
  src = SHAInet::SimpleMatrix.new(k, n)
  k.times { |i| n.times { |j| src[i, j] = ((i * n + j) % 17) * 0.125 - 1.0 } }
  SHAInet::Q4HostMatrix.from_simple(src)
end

def one_row(cols : Int32) : SHAInet::CudaMatrix
  x = SHAInet::CudaMatrix.new(1, cols)
  cols.times { |j| x[0, j] = (j % 5) * 0.25 - 0.5 }
  x.sync_to_device!("spec_in")
  x
end

describe "Q4HostMatrix promotion policy" do
  it "promotes on the first touch when the budget has headroom" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    k, n = 64, 128
    need = SHAInet::Q4CudaMatrix.device_bytes_for(k, n)
    # Room for many copies, so nothing can require eviction.
    with_budget(need * 64) do
      w = host_weight(k, n)
      x = one_row(k)

      before = SHAInet::Q4HostMatrix.cache_stats[:resident]
      w.gemv(x)
      after = SHAInet::Q4HostMatrix.cache_stats[:resident]

      # The whole point: ONE touch is now enough.
      (after - before).should eq(1)
    end
  end

  it "serves the second touch from the cache, so a repeat costs no transfer" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    k, n = 64, 128
    need = SHAInet::Q4CudaMatrix.device_bytes_for(k, n)
    with_budget(need * 64) do
      w = host_weight(k, n)
      x = one_row(k)

      w.gemv(x)
      mid = SHAInet::Q4HostMatrix.cache_stats
      w.gemv(x)
      done = SHAInet::Q4HostMatrix.cache_stats

      (done[:hits] - mid[:hits]).should eq(1)
      (done[:misses] - mid[:misses]).should eq(0)
    end
  end

  it "still withholds promotion from a one-off when the budget is full" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    k, n = 64, 128
    need = SHAInet::Q4CudaMatrix.device_bytes_for(k, n)
    # Exactly two copies fit, so all three weights are the SAME shape -- a larger third
    # would change the budget arithmetic and test nothing about policy. They are distinct
    # objects, and the cache keys on identity, so equal shapes stay separate entries.
    with_budget(need * 2) do
      hot_a = host_weight(k, n)
      hot_b = host_weight(k, n)
      x_a = one_row(k)
      x_b = one_row(k)

      # Fill the budget and keep both hot.
      2.times { hot_a.gemv(x_a) }
      2.times { hot_b.gemv(x_b) }
      filled = SHAInet::Q4HostMatrix.cache_stats
      filled[:resident].should eq(2)

      # A cold weight, touched once, must NOT displace either hot resident.
      cold = host_weight(k, n)
      cold.gemv(one_row(k))
      SHAInet::Q4HostMatrix.cache_stats[:resident].should eq(2)

      # And the hot pair is still resident: their next touches are hits, not re-fetches.
      before = SHAInet::Q4HostMatrix.cache_stats
      hot_a.gemv(x_a)
      hot_b.gemv(x_b)
      after = SHAInet::Q4HostMatrix.cache_stats
      (after[:hits] - before[:hits]).should eq(2)
      (after[:misses] - before[:misses]).should eq(0)
    end
  end

  it "keeps a weight larger than the whole budget off the cache entirely" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    k, n = 64, 128
    need = SHAInet::Q4CudaMatrix.device_bytes_for(k, n)
    # Budget below one copy: promotion can never succeed, and the streaming path must carry
    # it without raising or wedging.
    with_budget(need // 2) do
      w = host_weight(k, n)
      x = one_row(k)

      3.times { w.gemv(x) }

      SHAInet::Q4HostMatrix.cache_stats[:resident].should eq(0)
    end
  end

  it "refuses first-touch promotion when it would eat into the device reserve" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    k, n = 64, 128
    need = SHAInet::Q4CudaMatrix.device_bytes_for(k, n)
    saved_reserve = SHAInet::Q4HostMatrix.reserve_bytes
    begin
      # A reserve larger than the whole card: no promotion can ever satisfy it, so the
      # streaming path must carry the weight even though the BUDGET has ample room.
      # This is the guard that stops a generous budget consuming the VRAM that the KV cache
      # and workspaces need -- exactly the failure a 12000 MB budget produced on a 16 GB
      # card, where the cache filled and the next allocation died with cudaMalloc result 2.
      SHAInet::Q4HostMatrix.reserve_bytes = 1_u64 << 60
      with_budget(need * 64) do
        w = host_weight(k, n)
        x = one_row(k)

        3.times { w.gemv(x) }

        # Budget says yes, device reserve says no, and the reserve wins.
        SHAInet::Q4HostMatrix.cache_stats[:resident].should eq(0)
      end
    ensure
      SHAInet::Q4HostMatrix.reserve_bytes = saved_reserve
    end
  end

  it "promotes on first touch once the reserve is satisfiable" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    k, n = 64, 128
    need = SHAInet::Q4CudaMatrix.device_bytes_for(k, n)
    saved_reserve = SHAInet::Q4HostMatrix.reserve_bytes
    begin
      # The other direction of the same guard: with the reserve off, the identical weight and
      # budget DO promote, so the example above is proving the reserve and not something else.
      SHAInet::Q4HostMatrix.reserve_bytes = 0_u64
      with_budget(need * 64) do
        w = host_weight(k, n)
        w.gemv(one_row(k))
        SHAInet::Q4HostMatrix.cache_stats[:resident].should eq(1)
      end
    ensure
      SHAInet::Q4HostMatrix.reserve_bytes = saved_reserve
    end
  end

  it "produces the same result on the promoted path as on the streaming path" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?

    k, n = 64, 128
    need = SHAInet::Q4CudaMatrix.device_bytes_for(k, n)
    x = one_row(k)

    # Streaming: budget too small to ever promote.
    streamed = nil
    with_budget(need // 2) do
      w = host_weight(k, n)
      r = w.gemv(x)
      r.sync_from_device!("spec_stream")
      streamed = (0...n).map { |j| r[0, j] }
    end

    # Promoted on first touch.
    promoted = nil
    with_budget(need * 8) do
      w = host_weight(k, n)
      r = w.gemv(x)
      r.sync_from_device!("spec_promote")
      promoted = (0...n).map { |j| r[0, j] }
    end

    s = streamed.not_nil!
    p = promoted.not_nil!
    p.size.should eq(s.size)
    # Same weights, same input, same kernel: only the residency differs, so this is exact.
    s.each_with_index { |v, i| p[i].should be_close(v, 1e-6) }
  end
end
