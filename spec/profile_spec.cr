require "./spec_helper"

# The profiler exists to attribute decode time, so its own correctness matters:
# a phase that silently records nothing, or a `measure` that swallows the block's
# return value, would send the next optimization at the wrong target.
private def with_profile(enabled : Bool?, &)
  prev = SHAInet::Profile.enabled?
  SHAInet::Profile.enabled = enabled
  SHAInet::Profile.reset
  begin
    yield
  ensure
    SHAInet::Profile.enabled = prev
    SHAInet::Profile.reset
  end
end

private def with_env(key : String, value : String?, &)
  prev = ENV[key]?
  if value
    ENV[key] = value
  else
    ENV.delete(key)
  end
  begin
    yield
  ensure
    if prev
      ENV[key] = prev
    else
      ENV.delete(key)
    end
  end
end

describe SHAInet::Profile do
  describe "when disabled" do
    it "still returns the block's value" do
      with_profile(false) do
        SHAInet::Profile.measure("x") { 41 + 1 }.should eq 42
      end
    end

    it "records nothing at all" do
      with_profile(false) do
        100.times { SHAInet::Profile.measure("x") { 1 } }
        SHAInet::Profile.stats.should be_empty
        SHAInet::Profile.total_ns.should eq 0
      end
    end
  end

  describe "when enabled" do
    it "returns the block's value" do
      with_profile(true) do
        SHAInet::Profile.measure("x") { "value" }.should eq "value"
      end
    end

    it "counts every call and accumulates time" do
      with_profile(true) do
        5.times { SHAInet::Profile.measure("phase.a") { } }
        2.times { SHAInet::Profile.measure("phase.b") { } }
        stats = SHAInet::Profile.stats
        stats["phase.a"][:count].should eq 5
        stats["phase.b"][:count].should eq 2
        stats["phase.a"][:ns].should be >= 0
        SHAInet::Profile.total_ns.should eq(stats["phase.a"][:ns] + stats["phase.b"][:ns])
      end
    end

    it "measures a real duration rather than reporting zero" do
      with_profile(true) do
        SHAInet::Profile.measure("sleeper") { sleep 20.milliseconds }
        # Generous lower bound: the point is that the clock is actually read, not
        # that the scheduler is precise.
        SHAInet::Profile.stats["sleeper"][:ns].should be > 5_000_000
      end
    end

    it "still records when the block raises, and lets the error through" do
      with_profile(true) do
        expect_raises(ArgumentError) do
          SHAInet::Profile.measure("boom") { raise ArgumentError.new("nope") }
        end
        # An ensure-less implementation would lose the phase whenever the hot path
        # raised, which is exactly when you want the timing.
        SHAInet::Profile.stats["boom"][:count].should eq 1
      end
    end

    it "preserves first-seen order in stats" do
      with_profile(true) do
        SHAInet::Profile.measure("first") { }
        SHAInet::Profile.measure("second") { }
        SHAInet::Profile.measure("first") { }
        SHAInet::Profile.stats.keys.should eq ["first", "second"]
      end
    end

    it "does not lose records across concurrent fibers" do
      with_profile(true) do
        done = Channel(Nil).new
        4.times do
          spawn do
            250.times { SHAInet::Profile.record("shared", 1_i64) }
            done.send(nil)
          end
        end
        4.times { done.receive }
        SHAInet::Profile.stats["shared"][:count].should eq 1000
        SHAInet::Profile.stats["shared"][:ns].should eq 1000
      end
    end
  end

  describe ".reset" do
    it "clears totals, counts and order" do
      with_profile(true) do
        SHAInet::Profile.measure("x") { }
        SHAInet::Profile.reset
        SHAInet::Profile.stats.should be_empty
        SHAInet::Profile.total_ns.should eq 0
      end
    end
  end

  describe ".report" do
    it "lists phases slowest first with a total line" do
      with_profile(true) do
        SHAInet::Profile.record("slow", 5_000_000_i64)
        SHAInet::Profile.record("fast", 1_000_000_i64)
        io = IO::Memory.new
        SHAInet::Profile.report(io)
        text = io.to_s
        text.should contain "slow"
        text.should contain "fast"
        text.should contain "TOTAL"
        slow_at = text.index("slow") || -1
        fast_at = text.index("fast") || -1
        slow_at.should be >= 0
        fast_at.should be >= 0
        slow_at.should be < fast_at
      end
    end

    it "divides by the step count so the column is per token" do
      with_profile(true) do
        SHAInet::Profile.record("p", 10_000_000_i64) # 10 ms
        io = IO::Memory.new
        SHAInet::Profile.report(io, 10)
        io.to_s.should contain "1.0" # 10 ms over 10 steps
      end
    end

    it "says so instead of printing an empty table" do
      with_profile(true) do
        io = IO::Memory.new
        SHAInet::Profile.report(io)
        io.to_s.should contain "nothing recorded"
      end
    end
  end

  describe "default state" do
    it "is off unless SHAINET_PROFILE=1" do
      with_env("SHAINET_PROFILE", nil) do
        SHAInet::Profile.enabled = nil # forget any memoized decision
        SHAInet::Profile.enabled?.should be_false
      end
      with_env("SHAINET_PROFILE", "0") do
        SHAInet::Profile.enabled = nil
        SHAInet::Profile.enabled?.should be_false
      end
      with_env("SHAINET_PROFILE", "1") do
        SHAInet::Profile.enabled = nil
        SHAInet::Profile.enabled?.should be_true
      end
    ensure
      SHAInet::Profile.enabled = false
      SHAInet::Profile.reset
    end
  end
end
