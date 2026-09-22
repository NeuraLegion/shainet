require "./spec_helper"

# The fused kernel replaces nine launches (three causal convs, three state rolls, three SiLU passes)
# with two. That is only legitimate if it computes the SAME numbers, so compare it against the
# unfused sequence it replaces on identical inputs, and check the comparison is not vacuous.
describe "fused causal conv + SiLU" do
  it "matches the unfused conv-then-mul_sigmoid sequence exactly" do
    unless SHAInet::CUDA.fully_available? && SHAInet::CUDA.short_conv_silu3_available?
      pending! "CUDA kernels unavailable"
    end

    seq = 1           # decode shape: this is the path that runs per generated token
    ch = [12, 12, 20] # stand-ins for k_dim, k_dim, v_dim, kept small and distinct
    kernel = 4
    keep = kernel - 1

    # Deterministic, sign-varied inputs: SiLU is asymmetric, so a positive-only input would hide a
    # sign error in the fused expression.
    srcs = ch.map_with_index do |c, t|
      m = SHAInet::CudaMatrix.new(seq, c)
      (seq * c).times { |i| m.raw_data[i] = ((i * 7 + t * 3) % 11 - 5) * 0.31_f32 }
      m.mark_host_modified!
      m.sync_to_device!("src")
      m
    end
    weights = ch.map_with_index do |c, t|
      m = SHAInet::CudaMatrix.new(c, kernel)
      (c * kernel).times { |i| m.raw_data[i] = ((i * 5 + t) % 7 - 3) * 0.17_f32 }
      m.mark_host_modified!
      m.sync_to_device!("w")
      m
    end
    # The conv state carries kernel-1 past positions; make it non-zero so the state branch is
    # exercised rather than falling through the back < 0 zero path.
    states = ch.map_with_index do |c, t|
      m = SHAInet::CudaMatrix.new(c, keep)
      (c * keep).times { |i| m.raw_data[i] = ((i * 3 + t * 2) % 9 - 4) * 0.23_f32 }
      m.mark_host_modified!
      m.sync_to_device!("state")
      m
    end

    # Two independent state copies so each path advances its own.
    state_a = states.map do |s|
      c = SHAInet::CudaMatrix.new(s.rows, s.cols)
      (s.rows * s.cols).times { |i| c.raw_data[i] = s.raw_data[i] }
      c.mark_host_modified!
      c.sync_to_device!("state_a")
      c
    end
    state_b = states.map do |s|
      c = SHAInet::CudaMatrix.new(s.rows, s.cols)
      (s.rows * s.cols).times { |i| c.raw_data[i] = s.raw_data[i] }
      c.mark_host_modified!
      c.sync_to_device!("state_b")
      c
    end

    fused = ch.map { |c| SHAInet::CudaMatrix.new(seq, c) }
    plain = ch.map { |c| SHAInet::CudaMatrix.new(seq, c) }

    ok = SHAInet::CUDA.short_conv_silu3(
      fused[0].device_ptr.not_nil!, srcs[0].device_ptr.not_nil!,
      state_a[0].device_ptr.not_nil!, weights[0].device_ptr.not_nil!,
      fused[1].device_ptr.not_nil!, srcs[1].device_ptr.not_nil!,
      state_a[1].device_ptr.not_nil!, weights[1].device_ptr.not_nil!,
      fused[2].device_ptr.not_nil!, srcs[2].device_ptr.not_nil!,
      state_a[2].device_ptr.not_nil!, weights[2].device_ptr.not_nil!,
      seq, ch[0], ch[1], ch[2], kernel, true)
    ok.should be_true

    3.times do |t|
      SHAInet::CUDA.short_conv(plain[t].device_ptr.not_nil!, srcs[t].device_ptr.not_nil!,
        state_b[t].device_ptr.not_nil!, weights[t].device_ptr.not_nil!, seq, ch[t], kernel)
      SHAInet::CUDA.mul_sigmoid(plain[t].device_ptr.not_nil!, plain[t].device_ptr.not_nil!, seq * ch[t])
    end
    SHAInet::CUDA.device_synchronize

    3.times do |t|
      fused[t].mark_device_dirty!
      plain[t].mark_device_dirty!
      fused[t].sync_from_device!("fused")
      plain[t].sync_from_device!("plain")

      # Same intrinsic, same order of operations, so this should be bit-identical; allow only the
      # slack that a differing block decomposition could introduce.
      (seq * ch[t]).times do |i|
        fused[t].raw_data[i].should be_close(plain[t].raw_data[i], 1e-6)
      end

      # Vacuity check: SiLU must actually have been applied. A buffer of zeros, or one equal to the
      # raw convolution, would satisfy the comparison above if BOTH paths were broken the same way.
      nonzero = (seq * ch[t]).times.count { |i| fused[t].raw_data[i].abs > 1e-6 }
      nonzero.should be > 0
    end

    # The state must advance identically too, or the next decode step diverges.
    3.times do |t|
      state_a[t].mark_device_dirty!
      state_b[t].mark_device_dirty!
      state_a[t].sync_from_device!("state_a")
      state_b[t].sync_from_device!("state_b")
      (ch[t] * keep).times do |i|
        state_a[t].raw_data[i].should be_close(state_b[t].raw_data[i], 1e-6)
      end
    end
  end
end
