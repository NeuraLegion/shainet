require "../src/shainet"
require "json"

# Agentic chat demo: a multi-turn conversation with a growing context and a
# tool-calling loop, built entirely on Network#run. Think of it as a very light
# Claude/Copilot/Kiro-style CLI agent.
#
# The model (Qwen3-Coder-style: ChatML + XML tool calls) can call local tools;
# this harness parses the calls, runs the tools, feeds the results back, and
# lets the model continue until it produces a plain answer.
#
# Usage:
#   SHAINET_Q4=1 SHAINET_MOE_OFFLOAD=1 \
#     crystal run examples/agent.cr --release -Denable_cuda -- ~/models/Qwen3-Coder-30B-A3B-Instruct
#
# Built-in tools are intentionally read-only (list_directory, read_file). Add
# your own in build_tools — that is the extension point for an MCP bridge etc.

module AgentDemo
  MAX_READ_BYTES = 64_000
  MAX_TOOL_STEPS =      8

  record ToolParam, name : String, type : String, description : String, required : Bool = true
  record Message, role : String, content : String
  record ToolCall, name : String, args : Hash(String, String)

  class Tool
    getter name : String
    getter description : String
    getter params : Array(ToolParam)
    getter handler : Hash(String, String) -> String

    def initialize(@name, @description, @params, &@handler : Hash(String, String) -> String)
    end

    def call(args : Hash(String, String)) : String
      handler.call(args)
    end
  end

  # Read-only starter tools. Add more here (write_file, run_command, MCP, ...).
  def self.build_tools : Array(Tool)
    [
      Tool.new(
        "list_directory",
        "List the files and subdirectories in a directory.",
        [ToolParam.new("path", "string", "Directory path to list.")]
      ) do |args|
        path = args["path"]? || "."
        if Dir.exists?(path)
          Dir.children(path).sort.map { |e| Dir.exists?(File.join(path, e)) ? "#{e}/" : e }.join("\n")
        else
          "Error: not a directory: #{path}"
        end
      end,
      Tool.new(
        "read_file",
        "Read the contents of a text file (truncated if very large).",
        [ToolParam.new("path", "string", "File path to read.")]
      ) do |args|
        path = args["path"]? || ""
        if path.empty? || Dir.exists?(path) || !File.exists?(path)
          "Error: not a file: #{path}"
        else
          c = File.read(path)
          c.bytesize > MAX_READ_BYTES ? "#{c.byte_slice(0, MAX_READ_BYTES)}\n... [truncated]" : c
        end
      end,
    ]
  end

  # Extract <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call> blocks.
  def self.parse_tool_calls(text : String) : Array(ToolCall)
    calls = [] of ToolCall
    text.scan(/<tool_call>(.*?)<\/tool_call>/m) do |m|
      body = m[1]
      fmatch = body.match(/<function=([^>\s]+)>(.*)/m)
      next unless fmatch
      args = {} of String => String
      fmatch[2].scan(/<parameter=([^>\s]+)>\n?(.*?)\n?<\/parameter>/m) do |pm|
        args[pm[1].strip] = pm[2]
      end
      calls << ToolCall.new(fmatch[1].strip, args)
    end
    calls
  end

  class Agent
    @im_start : Int32
    @im_end : Int32
    @nl : Array(Int32)
    @stop_ids : Array(Int32)
    @system_block : String
    @messages : Array(Message)
    @sampler : SHAInet::Sampler
    getter max_context : Int32

    def initialize(@net : SHAInet::Network, @tokenizer : SHAInet::BPETokenizer, @tools : Array(Tool), @max_context : Int32 = 8192)
      im_start = @tokenizer.vocab["<|im_start|>"]?
      im_end = @tokenizer.vocab["<|im_end|>"]?
      raise "model is not ChatML (<|im_start|>/<|im_end|> missing); this agent targets Qwen3-style models" unless im_start && im_end
      @im_start = im_start.not_nil!
      @im_end = im_end.not_nil!
      @nl = @tokenizer.encode("\n")
      @stop_ids = [@im_end]
      ["<|endoftext|>", "<|end_of_text|>"].each { |n| (id = @tokenizer.vocab[n]?) && @stop_ids << id }
      @system_block = AgentDemo.render_tools_block(@tools)
      @sampler = SHAInet::Sampler.new(temperature: 0.3, top_k: 20, repetition_penalty: 1.1)
      @messages = [] of Message
    end

    # <|im_start|>{role}\n{content}<|im_end|>\n
    private def render_message(role : String, content : String) : Array(Int32)
      ids = [@im_start]
      ids.concat(@tokenizer.encode("#{role}\n#{content}"))
      ids << @im_end
      ids.concat(@nl)
      ids
    end

    private def build_prompt : Array(Int32)
      ids = render_message("system", @system_block)
      @messages.each do |m|
        if m.role == "tool"
          ids.concat(render_message("user", "<tool_response>\n#{m.content}\n</tool_response>"))
        else
          ids.concat(render_message(m.role, m.content))
        end
      end
      ids << @im_start
      ids.concat(@tokenizer.encode("assistant\n"))
      ids
    end

    # Current prompt size in tokens (full transcript + assistant primer).
    def context_tokens : Int32
      build_prompt.size
    end

    def reset
      @messages.clear
    end

    # Drop oldest messages (sliding window) until the prompt fits `target`
    # tokens, leaving a marker so the model knows history was trimmed. Keeps the
    # most recent messages (and never trims below one). Returns tokens removed.
    def compact!(target : Int32) : Int32
      before = context_tokens
      trimmed = false
      while context_tokens > target && @messages.size > 1
        @messages.shift
        trimmed = true
      end
      if trimmed && (@messages.empty? || @messages.first.content != TRUNCATION_MARKER)
        @messages.unshift(Message.new("user", TRUNCATION_MARKER))
      end
      before - context_tokens
    end

    TRUNCATION_MARKER = "[Earlier conversation was truncated to fit the context window.]"

    # One-line status: context usage, VRAM, expert-cache hit rate.
    def status : String
      parts = ["ctx #{context_tokens}/#{@max_context} tok"]
      if info = SHAInet::CUDA.memory_info
        used = info[:total] - info[:free]
        mb = 1024.0 * 1024.0
        parts << "VRAM #{(used / mb).round}/#{(info[:total] / mb).round} MB (#{(100.0 * used / info[:total]).round}%)"
      end
      if ENV.fetch("SHAINET_MOE_OFFLOAD", "0") == "1" && SHAInet::CUDA.fully_available?
        cs = SHAInet::Q4HostMatrix.cache_stats
        parts << "cache #{(cs[:hit_rate] * 100).round}% hit"
      end
      "[#{parts.join(" · ")}]"
    end

    # Generate one assistant turn from the current transcript. Re-prefills the
    # whole (growing) context each call — simple and correct for a demo.
    private def generate(max_tokens : Int32) : String
      @net.clear_cache!
      logits = @net.run(build_prompt, stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
      generated = [] of Int32
      max_tokens.times do
        row = logits.rows - 1
        @sampler.apply_repetition_penalty!(logits, generated, window: 20, row: row)
        id = @sampler.sample(logits, row)
        break if id < 0 || @stop_ids.includes?(id)
        break unless logits[row, id].finite?
        generated << id
        logits = @net.run([id], stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
      end
      @tokenizer.decode(generated)
    end

    # Handle one user input: run the tool loop until the model answers plainly.
    def chat(input : String, max_tokens : Int32)
      @messages << Message.new("user", input)
      step = 0
      loop do
        step += 1
        if step > MAX_TOOL_STEPS
          STDERR.puts "[agent] max tool steps reached; stopping"
          break
        end

        # Keep the prompt within the context budget (compact to 75% to leave room
        # for the response + any tool round-trips before the next compaction).
        if context_tokens > @max_context
          removed = compact!((@max_context * 0.75).to_i)
          STDERR.puts "  [agent] context compacted (−#{removed} tokens)" if removed > 0
        end

        text = generate(max_tokens)
        calls = AgentDemo.parse_tool_calls(text)

        if calls.empty?
          answer = text.strip
          @messages << Message.new("assistant", answer)
          puts "\nAgent: #{answer}"
          break
        end

        # Keep the assistant's tool_call markup verbatim in the transcript.
        @messages << Message.new("assistant", text.strip)
        calls.each do |c|
          STDERR.puts "  [tool] #{c.name}(#{c.args.map { |k, v| "#{k}=#{v.inspect}" }.join(", ")})"
          tool = @tools.find { |t| t.name == c.name }
          result =
            if tool
              begin
                tool.call(c.args)
              rescue ex
                "Error running #{c.name}: #{ex.message}"
              end
            else
              "Error: unknown tool #{c.name}"
            end
          @messages << Message.new("tool", result)
        end
      end
    end
  end

  # System prompt describing the tools in the Qwen3-Coder XML format.
  def self.render_tools_block(tools : Array(Tool)) : String
    String.build do |s|
      s << "You are a helpful AI coding assistant that can interact with the user's"
      s << " computer to solve tasks.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>"
      tools.each do |t|
        s << "\n<function>\n<name>" << t.name << "</name>"
        s << "\n<description>" << t.description << "</description>\n<parameters>"
        t.params.each do |p|
          s << "\n<parameter>\n<name>" << p.name << "</name>\n<type>" << p.type
          s << "</type>\n<description>" << p.description << "</description>\n</parameter>"
        end
        s << "\n</parameters>\n</function>"
      end
      s << "\n</tools>"
      s << "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
      s << "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
      s << "</parameter>\n</function>\n</tool_call>\n\n"
      s << "<IMPORTANT>\n- Function calls MUST be wrapped in <tool_call></tool_call> with an inner"
      s << " <function=...></function> block.\n- Provide any reasoning BEFORE the call, never after.\n"
      s << "- If no function is needed, just answer normally.\n</IMPORTANT>"
    end
  end
end

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
model_dir = ARGV[0]?
max_tokens = (ARGV[1]? || "512").to_i
unless model_dir && Dir.exists?(model_dir)
  STDERR.puts "Usage: agent <model-dir> [max_tokens]"
  STDERR.puts "  (point at an already-downloaded model, e.g. ~/models/Qwen3-Coder-30B-A3B-Instruct)"
  exit 1
end

STDERR.puts "Loading model from #{model_dir}..."
t0 = Time.instant
quantize = SHAInet::CUDA.fully_available? && !ENV["SHAINET_FP32"]?
bits = ENV["SHAINET_Q4"]? ? 4 : 8
offload = ENV.fetch("SHAINET_MOE_OFFLOAD", "0") == "1"
STDERR.puts "  Mode: #{ENV["SHAINET_FP32"]? ? "fp32" : "Q#{bits}"}#{offload ? " (MoE offload)" : ""}"
net = SHAInet::HFLoader.load(model_dir, quantize: quantize, bits: bits)
net.use_kv_cache = true
tokenizer = SHAInet::BPETokenizer.from_hf(File.join(model_dir, "tokenizer.json"))
STDERR.puts "Loaded in #{(Time.instant - t0).total_seconds.round(1)}s (vocab #{tokenizer.vocab.size})"

# The expert cache defaults to 70% of free VRAM, which is tuned for short, fixed
# prompts. An agent re-prefills a growing transcript each step, so leave a larger
# headroom (6GB) for context + activations to avoid OOM. Override via the env var.
if offload && !ENV["SHAINET_EXPERT_CACHE_MB"]? && (info = SHAInet::CUDA.memory_info)
  reserve = 6_u64 * 1024 * 1024 * 1024
  free = info[:free]
  budget_mb = free > reserve ? ((free - reserve) // (1024_u64 * 1024_u64)) : 0_u64
  ENV["SHAINET_EXPERT_CACHE_MB"] = budget_mb.to_s
  STDERR.puts "  Expert cache budget: #{budget_mb} MB (free #{free // (1024*1024)} MB − 6GB reserve)"
end

max_context = (ENV["SHAINET_AGENT_CONTEXT"]? || "8192").to_i
agent = AgentDemo::Agent.new(net, tokenizer, AgentDemo.build_tools, max_context)
STDERR.puts "Agent ready. Tools: #{AgentDemo.build_tools.map(&.name).join(", ")}."
STDERR.puts "Commands: /context  /compact  /clear  /help  (Ctrl-D to exit). Max context: #{max_context} tok."

loop do
  STDERR.print "\nYou: "
  input = gets
  break if input.nil?
  input = input.strip
  next if input.empty?

  case input
  when "/help"
    STDERR.puts "  /context  show context size, VRAM and cache usage"
    STDERR.puts "  /compact  trim the conversation history now"
    STDERR.puts "  /clear    reset the conversation"
    STDERR.puts "  /help     this message"
    next
  when "/context"
    STDERR.puts "  #{agent.status}"
    next
  when "/compact"
    removed = agent.compact!((agent.max_context * 0.75).to_i)
    STDERR.puts "  compacted −#{removed} tokens · #{agent.status}"
    next
  when "/clear"
    agent.reset
    STDERR.puts "  conversation cleared"
    next
  end

  agent.chat(input, max_tokens)
  STDERR.puts "  #{agent.status}"
end
STDERR.puts "\nbye"
