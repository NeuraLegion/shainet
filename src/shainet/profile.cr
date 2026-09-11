module SHAInet
  # Phase timer for the inference hot path.
  #
  # Exists to answer "where does a decode step actually go?" with measurements
  # instead of arithmetic. A 30B-A3B decode step measured ~130 ms while the summed
  # GEMV time was ~12 ms, so ~90% was unaccounted for; guessing at that from the
  # call graph is how you end up optimizing the 10%.
  #
  # Off by default and compiled in unconditionally. `measure` takes a block and
  # `yield`s it, so when profiling is off the whole call inlines to the block plus
  # one boolean test. Enable with SHAINET_PROFILE=1 or `Profile.enabled = true`.
  #
  # Phases are deliberately LEAVES that partition the work: nothing here wraps
  # another phase, so the recorded times sum to roughly the whole step and the
  # percentages mean what they look like. Adding a phase that contains an existing
  # one would double count.
  module Profile
    @@enabled : Bool? = nil
    @@mutex = Mutex.new
    @@totals = Hash(String, Int64).new(0_i64)
    @@counts = Hash(String, Int64).new(0_i64)
    @@order = [] of String

    def self.enabled? : Bool
      flag = @@enabled
      return flag unless flag.nil?
      flag = ENV.fetch("SHAINET_PROFILE", "0") == "1"
      @@enabled = flag
      flag
    end

    # Force profiling on or off. Pass nil to forget the decision and re-read
    # SHAINET_PROFILE on the next query.
    def self.enabled=(value : Bool?)
      @@enabled = value
    end

    # Time a phase. Returns the block's value either way.
    def self.measure(phase : String, &)
      return yield unless enabled?
      t0 = Time.monotonic
      begin
        yield
      ensure
        record(phase, (Time.monotonic - t0).total_nanoseconds.to_i64)
      end
    end

    # Record an already-measured duration, for call sites that cannot wrap a block.
    def self.record(phase : String, nanos : Int64) : Nil
      @@mutex.synchronize do
        @@order << phase unless @@counts.has_key?(phase)
        @@totals[phase] += nanos
        @@counts[phase] += 1
      end
    end

    def self.reset : Nil
      @@mutex.synchronize do
        @@totals.clear
        @@counts.clear
        @@order.clear
      end
    end

    # Snapshot of every phase seen so far.
    def self.stats : Hash(String, NamedTuple(ns: Int64, count: Int64))
      @@mutex.synchronize do
        out = Hash(String, NamedTuple(ns: Int64, count: Int64)).new
        @@order.each { |p| out[p] = {ns: @@totals[p], count: @@counts[p]} }
        out
      end
    end

    def self.total_ns : Int64
      @@mutex.synchronize { @@totals.values.sum(0_i64) }
    end

    # Human-readable breakdown, slowest phase first. `steps` divides the totals so
    # the per-step column is per token rather than per run.
    def self.report(io : IO = STDERR, steps : Int32 = 1) : Nil
      snap = stats
      return io.puts("profile: nothing recorded (SHAINET_PROFILE=1 to enable)") if snap.empty?
      total = snap.values.sum(0_i64) { |v| v[:ns] }
      total = 1_i64 if total == 0
      divisor = steps < 1 ? 1 : steps

      io.puts "phase                        calls     total ms   per step ms      %"
      io.puts "-" * 68
      snap.to_a.sort_by { |(_, v)| -v[:ns] }.each do |(phase, v)|
        ms = v[:ns] / 1_000_000.0
        io.puts "#{phase.ljust(26)} #{v[:count].to_s.rjust(8)} #{ms.round(2).to_s.rjust(12)} #{(ms / divisor).round(3).to_s.rjust(13)} #{(100.0 * v[:ns] / total).round(1).to_s.rjust(6)}"
      end
      io.puts "-" * 68
      io.puts "#{"TOTAL".ljust(26)} #{"".rjust(8)} #{(total / 1_000_000.0).round(2).to_s.rjust(12)} #{(total / 1_000_000.0 / divisor).round(3).to_s.rjust(13)} #{"100.0".rjust(6)}"
    end
  end
end
