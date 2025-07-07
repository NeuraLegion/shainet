require "wait_group"

module SHAInet
  # Actor-like wrapper around a tokenizer that encodes texts in parallel.
  #
  # It spawns a number of worker fibers that receive tasks over a channel.
  # When compiled with `-Dpreview_mt -Dexecution_context` the workers run on a
  # dedicated `ExecutionContext::MultiThreaded` allowing parallel execution.
  # Otherwise the same implementation works with Crystal's default scheduler.
  class ConcurrentTokenizer(T)
    private struct EncodeJob
      getter text : String
      getter result : Channel(Array(Int32))

      def initialize(@text : String, @result : Channel(Array(Int32)))
      end
    end

    alias Job = EncodeJob?

    @jobs : Channel(Job)
    @workers : Array(Fiber)
    @worker_count : Int32

    def initialize(@tokenizer : T, worker_count : Int32 = 4)
      @worker_count = worker_count
      @jobs = Channel(Job).new
      @workers = [] of Fiber

      {% if flag?(:execution_context) %}
        context = Fiber::ExecutionContext::MultiThreaded.new("tokenizer-workers", worker_count)
        worker_count.times do
          @workers << spawn(context: context) { worker_loop }
        end
      {% else %}
        worker_count.times do
          @workers << spawn { worker_loop }
        end
      {% end %}
    end

    # Encode a single text by sending it to a worker and waiting on the result.
    def encode(text : String) : Array(Int32)
      result = Channel(Array(Int32)).new
      @jobs.send EncodeJob.new(text, result)
      result.receive
    end

    # Encode multiple texts concurrently using `WaitGroup`.
    def encode_batch(texts : Array(String)) : Array(Array(Int32))
      results = Array(Array(Int32) | Nil).new(texts.size) { nil }

      WaitGroup.wait do |wg|
        texts.each_with_index do |text, idx|
          wg.spawn do
            results[idx] = encode(text)
          end
        end
      end

      results.compact
    end

    # Stop all workers. Pending jobs will be processed first.
    def shutdown
      @worker_count.times { @jobs.send(nil) }
    end

    private def worker_loop
      while job = @jobs.receive?
        result = @tokenizer.encode(job.text)
        job.result.send(result)
      end
    end
  end
end
