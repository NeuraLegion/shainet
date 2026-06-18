require "../src/shainet"
require "json"
require "colorize"

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

  # Confirmation gate for mutating/executing tools (skipped with SHAINET_AGENT_YOLO=1).
  def self.confirm?(desc : String) : Bool
    return true if ENV.fetch("SHAINET_AGENT_YOLO", "0") == "1"
    STDERR.print "  #{"⚠ allow".colorize(:yellow)} #{desc}? [y/N] "
    STDERR.flush
    ans = gets
    !!(ans && ans.strip.downcase.starts_with?("y"))
  end

  # Basic code-agent tools. Read-only ones run freely; write/edit/run ask for
  # confirmation first (unless SHAINET_AGENT_YOLO=1).
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
      Tool.new(
        "search",
        "Search files under a directory for a regular-expression pattern (grep-like). Returns file:line matches.",
        [ToolParam.new("pattern", "string", "Regular expression to search for."),
         ToolParam.new("path", "string", "Directory or file to search (default: current dir).", false)]
      ) do |args|
        pat = args["pattern"]? || ""
        next "Error: empty pattern" if pat.empty?
        root = args["path"]? || "."
        begin
          re = Regex.new(pat)
        rescue ex
          next "Error: invalid regex: #{ex.message}"
        end
        files = Dir.exists?(root) ? Dir.glob(File.join(root, "**", "*")) : [root]
        results = [] of String
        files.each do |f|
          break if results.size >= 100
          next if Dir.exists?(f) || !File.exists?(f) || File.size(f) > 2_000_000
          begin
            File.read_lines(f).each_with_index do |line, i|
              if re.matches?(line)
                results << "#{f}:#{i + 1}: #{line.strip}"
                break if results.size >= 100
              end
            end
          rescue
            # skip unreadable/binary files
          end
        end
        results.empty? ? "No matches." : results.join("\n")
      end,
      Tool.new(
        "write_file",
        "Create or overwrite a text file with the given content.",
        [ToolParam.new("path", "string", "File path to write."),
         ToolParam.new("content", "string", "Full file content.")]
      ) do |args|
        path = args["path"]? || ""
        next "Error: empty path" if path.empty?
        content = args["content"]? || ""
        next "Declined by user." unless AgentDemo.confirm?("write #{content.bytesize} bytes to #{path}")
        File.write(path, content)
        "Wrote #{content.bytesize} bytes to #{path}"
      end,
      Tool.new(
        "edit_file",
        "Replace exact text in a file. Replaces all literal occurrences of 'find' with 'replace'.",
        [ToolParam.new("path", "string", "File path to edit."),
         ToolParam.new("find", "string", "Exact text to find."),
         ToolParam.new("replace", "string", "Replacement text.")]
      ) do |args|
        path = args["path"]? || ""
        next "Error: not a file: #{path}" if path.empty? || Dir.exists?(path) || !File.exists?(path)
        find = args["find"]? || ""
        next "Error: empty 'find'" if find.empty?
        replace = args["replace"]? || ""
        content = File.read(path)
        count = content.scan(find).size
        next "No occurrences of the given text in #{path}" if count == 0
        next "Declined by user." unless AgentDemo.confirm?("replace #{count} occurrence(s) in #{path}")
        File.write(path, content.gsub(find, replace))
        "Replaced #{count} occurrence(s) in #{path}"
      end,
      Tool.new(
        "run_command",
        "Run a shell command and return its combined stdout/stderr and exit code.",
        [ToolParam.new("command", "string", "The shell command to execute.")]
      ) do |args|
        cmd = args["command"]? || ""
        next "Error: empty command" if cmd.empty?
        next "Declined by user." unless AgentDemo.confirm?("run: #{cmd}")
        buf = IO::Memory.new
        status = Process.run("/bin/sh", ["-c", cmd], output: buf, error: buf)
        result = buf.to_s
        result = "#{result.byte_slice(0, MAX_READ_BYTES)}\n... [truncated]" if result.bytesize > MAX_READ_BYTES
        "exit=#{status.exit_code}\n#{result}"
      end,
    ]
  end

  # Streams assistant output token-by-token, coloring reasoning (<think>…
  # </think>) dim and the answer normal, while hiding tool-call XML. Buffers a
  # small tail so tags split across tokens are never shown half-rendered.
  class StreamRenderer
    TAGS = {"<think>" => :think_open, "</think>" => :think_close,
            "<tool_call>" => :tool_open, "</tool_call>" => :tool_close}

    def initialize(@io : IO)
      @mode = :normal
      @pending = ""
      @started = false
    end

    def feed(text : String)
      @pending += text
      loop do
        idx = nil
        found = nil
        TAGS.each_key do |tag|
          i = @pending.index(tag)
          if i && (idx.nil? || i < idx.not_nil!)
            idx, found = i, tag
          end
        end
        if (i = idx) && (tag = found)
          emit(@pending[0...i])
          apply(TAGS[tag])
          @pending = @pending[(i + tag.size)..]
        else
          hold = partial_tag_suffix(@pending)
          emit(@pending[0, @pending.size - hold])
          @pending = @pending[(@pending.size - hold)..]
          break
        end
      end
    end

    def finish
      emit(@pending)
      @pending = ""
    end

    private def emit(t : String)
      return if t.empty?
      # Trim leading whitespace before the very first visible chars.
      unless @started
        t = t.lstrip
        return if t.empty?
        @started = true
      end
      case @mode
      when :think  then @io.print t.colorize(:dark_gray)
      when :normal then @io.print t
      end # :tool -> hidden
      @io.flush
    end

    private def apply(action)
      case action
      when :think_open  then @mode = :think
      when :think_close then @mode = :normal
      when :tool_open   then @mode = :tool
      when :tool_close  then @mode = :normal
      end
    end

    # Longest suffix of s that is a (shorter) prefix of some tag — held back in
    # case the tag completes on the next token.
    private def partial_tag_suffix(s : String) : Int32
      max = 0
      TAGS.each_key do |tag|
        (1...tag.size).each do |k|
          max = k if k <= s.size && k > max && s[(s.size - k)..] == tag[0, k]
        end
      end
      max
    end
  end

  # Extract <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call> blocks.
  def self.parse_tool_calls(text : String) : Array(ToolCall)
    text = text.scrub # never run regex on invalid UTF-8 (broken-model output)
    calls = [] of ToolCall
    text.scan(/<tool_call>(.*?)<\/tool_call>/m) do |m|
      body = m[1]
      fmatch = body.match(/<function=([^>\s]+)>(.*)/m)
      next unless fmatch
      name = fmatch[1].strip
      next if name.empty? || name == "none" # ignore placeholder/hallucinated calls
      args = {} of String => String
      fmatch[2].scan(/<parameter=([^>\s]+)>\n?(.*?)\n?<\/parameter>/m) do |pm|
        args[pm[1].strip] = pm[2]
      end
      calls << ToolCall.new(name, args)
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

    def initialize(@net : SHAInet::Network, @tokenizer : SHAInet::BPETokenizer, @tools : Array(Tool), @max_context : Int32 = 4096)
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

    # Tokens one message contributes to the prompt.
    private def message_tokens(m : Message) : Int32
      if m.role == "tool"
        render_message("user", "<tool_response>\n#{m.content}\n</tool_response>").size
      else
        render_message(m.role, m.content).size
      end
    end

    # Summarize a slice of old messages into a single concise note using the
    # model itself (Claude/Kiro-style compaction) so history is condensed rather
    # than lost. Falls back to a plain marker if the model returns nothing.
    private def summarize(msgs : Array(Message)) : String
      transcript = String.build do |s|
        msgs.each { |m| s << m.role << ": " << m.content << "\n\n" }
      end
      instruction = "Summarize the following conversation between a user and an AI coding " \
                    "assistant. Preserve key facts, decisions, file paths, tool results, and " \
                    "any unfinished tasks. Be concise.\n\n#{transcript}"
      prompt = render_message("system", "You write concise, faithful conversation summaries.")
      prompt.concat(render_message("user", instruction))
      prompt << @im_start
      prompt.concat(@tokenizer.encode("assistant\n"))
      summary = generate_from(prompt, 384).strip
      summary.empty? ? TRUNCATION_MARKER : summary
    end

    # Compact the transcript to fit `target` tokens: keep a recent tail, and
    # replace the older head with a model-written summary. Hard-trims as a last
    # resort if it still doesn't fit. Returns tokens removed.
    def compact!(target : Int32) : Int32
      before = context_tokens
      return 0 if before <= target || @messages.size <= 1

      # Keep the most recent messages within ~half the target; summarize the rest.
      keep_budget = target // 2
      tail = [] of Message
      tail_tokens = 0
      @messages.reverse_each do |m|
        t = message_tokens(m)
        break if !tail.empty? && tail_tokens + t > keep_budget
        tail.unshift(m)
        tail_tokens += t
      end
      head = @messages[0, @messages.size - tail.size]

      if head.empty?
        # Nothing old enough to summarize — fall back to plain truncation.
        return truncate!(target)
      end

      summary = summarize(head)
      @messages = [Message.new("user", "[Summary of earlier conversation]\n#{summary}")] + tail
      # If the summary + tail still overflow, drop oldest until it fits.
      truncate!(target)
      before - context_tokens
    end

    # Plain sliding-window drop of oldest messages with a marker. Returns tokens removed.
    private def truncate!(target : Int32) : Int32
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

    # Generate from an explicit prompt; re-prefills it fresh (KV cleared). When
    # echo is set, the user-facing prose is streamed live (with <think> and
    # tool-call markup filtered out).
    private def generate_from(prompt : Array(Int32), max_tokens : Int32, echo : Bool = false) : String
      @net.clear_cache!
      logits = @net.run(prompt, stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
      generated = [] of Int32
      renderer = echo ? AgentDemo::StreamRenderer.new(STDERR) : nil
      prev = ""
      max_tokens.times do
        row = logits.rows - 1
        @sampler.apply_repetition_penalty!(logits, generated, window: 20, row: row)
        id = @sampler.sample(logits, row)
        break if id < 0 || @stop_ids.includes?(id)
        break unless logits[row, id].finite?
        generated << id
        if r = renderer
          full = @tokenizer.decode(generated).scrub
          r.feed(full[prev.size..]) if full.size > prev.size
          prev = full
        end
        logits = @net.run([id], stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
      end
      renderer.try(&.finish)
      # scrub: a broken model can emit tokens that decode to invalid UTF-8, which
      # would crash downstream regex/parsing.
      @tokenizer.decode(generated).scrub
    end

    # Generate one assistant turn from the current transcript, streaming the
    # filtered prose live. Re-prefills the whole (growing) context each call.
    private def generate(max_tokens : Int32) : String
      generate_from(build_prompt, max_tokens, echo: true)
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

        STDERR.print "\n#{"Agent".colorize(:light_cyan).bold} ❯ "
        STDERR.flush
        begin
          text = generate(max_tokens) # streams the filtered prose inline
        rescue ex
          STDERR.puts "\n  [agent] generation failed: #{ex.message}".colorize(:red)
          STDERR.puts "  (out of GPU memory? try /clear, a shorter request, or a smaller SHAINET_EXPERT_CACHE_MB)".colorize(:dark_gray)
          @net.clear_cache!
          break
        end
        STDERR.puts ""
        calls = AgentDemo.parse_tool_calls(text)

        # Keep the assistant's output (incl. any tool_call markup) verbatim.
        @messages << Message.new("assistant", text.strip)
        break if calls.empty?

        calls.each do |c|
          STDERR.puts "  #{"⚒ #{c.name}".colorize(:yellow)}(#{c.args.map { |k, v| "#{k}=#{v.inspect}" }.join(", ")})".colorize(:dark_gray)
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

  # ASCII splash banner.
  def self.print_banner(io : IO, subtitle : String)
    # ameba:disable Style/HeredocEscape
    art = <<-'ASCII'
       ____  _   _    _    ___            _
      / ___|| | | |  / \  |_ _|_ __   ___| |_
      \___ \| |_| | / _ \  | || '_ \ / _ \ __|
       ___) |  _  |/ ___ \ | || | | |  __/ |_
      |____/|_| |_/_/   \_\___|_| |_|\___|\__|
      ASCII
    io.puts
    io.puts art.colorize(:light_cyan).bold
    io.puts "  ▸ Agent ".colorize(:cyan).bold.to_s + "· #{subtitle}".colorize(:dark_gray).to_s
    io.puts
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

Colorize.enabled = STDERR.tty?
STDERR.sync = true # stream tokens as they arrive (no buffering)
AgentDemo.print_banner(STDERR, "#{File.basename(model_dir)} · local coding agent on Network#run")

STDERR.puts "Loading model from #{model_dir}...".colorize(:dark_gray)
t0 = Time.instant
quantize = SHAInet::CUDA.fully_available? && !ENV["SHAINET_FP32"]?
bits = ENV["SHAINET_Q4"]? ? 4 : 8
offload = ENV.fetch("SHAINET_MOE_OFFLOAD", "0") == "1"
STDERR.puts "  Mode: #{ENV["SHAINET_FP32"]? ? "fp32" : "Q#{bits}"}#{offload ? " (MoE offload)" : ""}"
net = SHAInet::HFLoader.load(model_dir, quantize: quantize, bits: bits)
net.use_kv_cache = true
tokenizer = SHAInet::BPETokenizer.from_hf(File.join(model_dir, "tokenizer.json"))
STDERR.puts "Loaded in #{(Time.instant - t0).total_seconds.round(1)}s (vocab #{tokenizer.vocab.size})"

# Size the expert cache to leave headroom for the model + prefill activations.
# (cudaMalloc now GC-reclaims dead GPU buffers on pressure, so this only needs a
# modest reserve.) Override with SHAINET_EXPERT_CACHE_MB (0 disables).
if offload && !ENV["SHAINET_EXPERT_CACHE_MB"]? && (info = SHAInet::CUDA.memory_info)
  reserve = 6_u64 * 1024 * 1024 * 1024
  free = info[:free]
  budget_mb = free > reserve ? ((free - reserve) // (1024_u64 * 1024_u64)) : 0_u64
  ENV["SHAINET_EXPERT_CACHE_MB"] = budget_mb.to_s
  STDERR.puts "  Expert cache budget: #{budget_mb} MB (free #{free // (1024*1024)} MB − 6GB reserve)".colorize(:dark_gray)
end

max_context = (ENV["SHAINET_AGENT_CONTEXT"]? || "4096").to_i
agent = AgentDemo::Agent.new(net, tokenizer, AgentDemo.build_tools, max_context)
STDERR.puts "Ready · tools: #{AgentDemo.build_tools.map(&.name).join(", ")} · max context #{max_context} tok".colorize(:green)
STDERR.puts "Commands: /context  /compact  /clear  /help   (Ctrl-D to exit)".colorize(:dark_gray)

loop do
  STDERR.print "\n#{"You".colorize(:light_green).bold} ❯ "
  input = gets
  break if input.nil?
  input = input.strip
  next if input.empty?

  case input
  when "/help"
    STDERR.puts "  /context  show context size, VRAM and cache usage".colorize(:dark_gray)
    STDERR.puts "  /compact  summarize + trim the conversation history now".colorize(:dark_gray)
    STDERR.puts "  /clear    reset the conversation".colorize(:dark_gray)
    STDERR.puts "  /help     this message".colorize(:dark_gray)
    next
  when "/context"
    STDERR.puts "  #{agent.status}".colorize(:dark_gray)
    next
  when "/compact"
    removed = agent.compact!((agent.max_context * 0.75).to_i)
    STDERR.puts "  compacted −#{removed} tokens · #{agent.status}".colorize(:dark_gray)
    next
  when "/clear"
    agent.reset
    STDERR.puts "  conversation cleared".colorize(:dark_gray)
    next
  end

  agent.chat(input, max_tokens)
  STDERR.puts "  #{agent.status}".colorize(:dark_gray)
end
STDERR.puts "\nbye 👋".colorize(:cyan)
