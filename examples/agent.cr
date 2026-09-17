require "../src/shainet"
require "json"
require "colorize"
require "./agent_workspace"
require "./agent_v4a"

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

  # Tool-calling iterations allowed per user turn, before the agent hands control back.
  #
  # A runaway guard, not a work budget: without it a model that keeps emitting tool calls
  # never returns to the prompt. The number was 8, chosen when a turn's prefill cost minutes
  # and a runaway was expensive. Prefill is now ~29 s at 4096 with KV prefix reuse making
  # turn 2+ about a second, so 8 stopped being a runaway guard and started truncating
  # ordinary work -- reading four files and grepping twice already exceeds it.
  #
  # 40 is high enough that hitting it means something is actually looping.
  # AGENT_MAX_TOOL_STEPS overrides it, clamped to at least 1: a 0 would break out before
  # generating anything and every turn would answer with silence.
  MAX_TOOL_STEPS = begin
    n = (ENV["AGENT_MAX_TOOL_STEPS"]? || "40").to_i
    n < 1 ? 1 : n
  end

  record ToolParam, name : String, type : String, description : String, required : Bool = true
  # `ids` holds the EXACT tokens the model produced for an assistant turn, when they are
  # known. Re-encoding the decoded text does not reliably reproduce them -- a decode,
  # UTF-8 scrub and re-encode round trip can shift token boundaries -- and a single
  # mismatched token destroys KV prefix reuse for the whole conversation.
  record Message, role : String, content : String, ids : Array(Int32)? = nil
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

  # Root that every file tool is confined to: the directory the agent was launched in, unless
  # AGENT_WORKSPACE says otherwise.
  #
  # Resolved once at startup rather than per call, so nothing can widen its own sandbox by
  # chdir-ing partway through a turn.
  @@workspace = File.realpath(ENV["AGENT_WORKSPACE"]? || Dir.current)

  def self.workspace : String
    @@workspace
  end

  # Resolve a model-supplied path. Returns {path, ""} on success and {nil, error_text} on
  # refusal, so a tool body stays a single line and the model gets told WHY.
  #
  # Inside the workspace needs no approval. OUTSIDE it, the path is negotiable rather than
  # forbidden: the user is asked once per location and the answer is remembered, so working on
  # a file elsewhere costs one prompt instead of one per tool call. An approval is per PATH on
  # purpose -- approving a model config should not also hand over ~/.ssh/id_rsa. Answering "a"
  # still opens everything, matching the prompt's existing meaning.
  def self.safe_path(raw : String?) : {String?, String}
    {WorkspacePath.resolve(@@workspace, raw || ""), ""}
  rescue WorkspaceEscapeError
    outside_path(raw || "")
  rescue ex : WorkspacePathError
    {nil, "Error: #{ex.message}."}
  end

  private def self.outside_path(raw : String) : {String?, String}
    resolved = WorkspacePath.resolve_anywhere(@@workspace, raw)
    return {resolved, ""} if allow_all? || WorkspacePath.approved_outside?(resolved)
    unless confirm?("access #{resolved} — OUTSIDE the workspace (#{@@workspace})")
      return {nil, "Error: #{raw} is outside the workspace and the user declined access. " \
                   "Work within #{@@workspace} instead."}
    end
    # Remembered, so the next tool call on this file does not ask again.
    WorkspacePath.approve_outside(resolved)
    {resolved, ""}
  rescue ex : WorkspacePathError
    {nil, "Error: #{ex.message}."}
  end

  # A unified-ish diff of a proposed whole-file write, for the confirmation prompt.
  #
  # Approving "write 4182 bytes to x.cr" says nothing about what actually changes, which is
  # the real problem with a blanket approval: allow-all is only reasonable if each prompt
  # showed enough to judge. Capped on purpose -- a 3000-line rewrite scrolling past defeats
  # the point as thoroughly as showing nothing.
  def self.preview_diff(path : String, new_content : String, max_lines : Int32 = 40) : String
    old_lines = File.exists?(path) ? File.read(path).lines : [] of String
    new_lines = new_content.lines
    # Trim the common head and tail, so the window shown is the change and not the file.
    head = 0
    while head < old_lines.size && head < new_lines.size && old_lines[head] == new_lines[head]
      head += 1
    end
    tail = 0
    while tail < (old_lines.size - head) && tail < (new_lines.size - head) &&
          old_lines[old_lines.size - 1 - tail] == new_lines[new_lines.size - 1 - tail]
      tail += 1
    end
    removed = old_lines[head, old_lines.size - head - tail]
    added = new_lines[head, new_lines.size - head - tail]
    return "  (no textual change)" if removed.empty? && added.empty?

    half = max_lines // 2
    out = [] of String
    out << "  @@ line #{head + 1} @@"
    removed.first(half).each { |l| out << "  -#{l}" }
    out << "  ... #{removed.size - half} more removed" if removed.size > half
    added.first(half).each { |l| out << "  +#{l}" }
    out << "  ... #{added.size - half} more added" if added.size > half
    out.join("\n")
  end

  # Set by answering "a" at a confirmation prompt: allow every later mutating tool call for
  # the rest of the process, without re-asking.
  #
  # Deliberately in-memory and process-scoped, not persisted: a blanket approval that
  # outlived the session would be a much bigger promise than the one the user made at the
  # prompt. SHAINET_AGENT_YOLO=1 is the same thing chosen up front.
  @@allow_all = false

  def self.allow_all? : Bool
    @@allow_all
  end

  def self.allow_all=(value : Bool)
    @@allow_all = value
  end

  # Confirmation gate for mutating/executing tools.
  #
  #   y  allow this call
  #   n  deny this call (the default, so a bare Enter is safe)
  #   a  allow this and every later call this session
  #
  # Skipped entirely with SHAINET_AGENT_YOLO=1.
  def self.confirm?(desc : String) : Bool
    return true if ENV.fetch("SHAINET_AGENT_YOLO", "0") == "1"
    return true if @@allow_all
    STDERR.print "  #{"⚠ allow".colorize(:yellow)} #{desc}? [y/N/a=all] "
    STDERR.flush
    ans = gets
    # EOF (piped input, closed stdin) must not read as approval.
    return false unless ans
    case ans.strip.downcase
    when "a", "all"
      @@allow_all = true
      STDERR.puts "  #{"⚠ approving all further tool calls this session".colorize(:yellow)}"
      true
    when .starts_with?("y")
      true
    else
      false
    end
  end

  # Basic code-agent tools. Read-only ones run freely; write/edit/run ask for
  # confirmation first (unless SHAINET_AGENT_YOLO=1).
  def self.build_tools : Array(Tool)
    [
      Tool.new(
        "list_directory",
        "List the files and subdirectories in a directory. Paths are relative to the workspace root; an absolute path outside it needs the user to approve that location once.",
        [ToolParam.new("path", "string", "Directory path to list (default: current directory).")]
      ) do |args|
        raw = (args["path"]? || "").strip
        raw = "." if raw.empty?
        path, err = AgentDemo.safe_path(raw)
        if path.nil?
          next err
        end
        if Dir.exists?(path)
          Dir.children(path).sort.map { |e| Dir.exists?(File.join(path, e)) ? "#{e}/" : e }.join("\n")
        else
          "Error: not a directory: #{raw}"
        end
      end,
      Tool.new(
        "read_file",
        "Read the contents of a text file (truncated if very large). Binary files are refused. Paths are relative to the workspace root; an absolute path outside it needs the user to approve that location once.",
        [ToolParam.new("path", "string", "File path to read.")]
      ) do |args|
        path, err = AgentDemo.safe_path(args["path"]?)
        if path.nil?
          next err
        end
        if Dir.exists?(path) || !File.exists?(path)
          "Error: not a file: #{args["path"]?}"
        else
          c = File.read(path)
          # Refuse binaries outright. A model asked to inspect a directory will happily try
          # to read a compiled executable, and the bytes are useless to it while being
          # ruinously expensive: 64 KB of binary tokenized to 60534 tokens in one observed
          # run, against an 8192-token window. A NUL byte is the cheap, reliable signal.
          if c.byte_slice(0, Math.min(c.bytesize, 8192)).includes?('\u0000')
            "Error: #{path} looks like a binary file (contains NUL bytes); refusing to read it"
          elsif !c.valid_encoding?
            "Error: #{path} is not valid UTF-8; refusing to read it"
          elsif c.bytesize > MAX_READ_BYTES
            "#{c.byte_slice(0, MAX_READ_BYTES)}\n... [truncated]"
          else
            c
          end
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
        root, err = AgentDemo.safe_path(args["path"]? || ".")
        if root.nil?
          next err
        end
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
                # Report paths relative to the workspace: absolute ones leak the host layout
                # into the transcript and are not what the model may pass back.
                rel = f.starts_with?("#{AgentDemo.workspace}/") ? f[(AgentDemo.workspace.size + 1)..] : f
                results << "#{rel}:#{i + 1}: #{line.strip}"
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
        "Create a NEW text file. An existing file is refused: use apply_patch or edit_file to change one. Paths are relative to the workspace root; an absolute path outside it needs the user to approve that location once.",
        [ToolParam.new("path", "string", "File path to write."),
         ToolParam.new("content", "string", "Full file content.")]
      ) do |args|
        path, err = AgentDemo.safe_path(args["path"]?)
        if path.nil?
          next err
        end
        # Refuse to clobber. A model that means to change three lines will cheerfully pass a
        # whole-file rewrite, and any part it did not remember is silently gone. Making it
        # patch or edit instead keeps the rest of the file out of the blast radius.
        if File.exists?(path)
          next "Error: #{args["path"]?} already exists. Use apply_patch or edit_file to modify it, " \
               "or delete it first with run_command if replacing it wholesale is really intended."
        end
        content = args["content"]? || ""
        rel = args["path"]?
        STDERR.puts AgentDemo.preview_diff(path, content) unless AgentDemo.allow_all?
        next "Declined by user." unless AgentDemo.confirm?("create #{rel} (#{content.bytesize} bytes)")
        # mkdir -p the parents: otherwise a new file in a new directory needs a shell call.
        Dir.mkdir_p(File.dirname(path))
        File.write(path, content)
        "Wrote #{content.bytesize} bytes to #{rel}"
      end,
      Tool.new(
        "edit_file",
        "Replace an exact string in a file. 'find' must match EXACTLY ONE occurrence, including whitespace; include surrounding context to make it unique. Paths are relative to the workspace root; an absolute path outside it needs the user to approve that location once.",
        [ToolParam.new("path", "string", "File path to edit."),
         ToolParam.new("find", "string", "Exact text to find (must be unique in the file)."),
         ToolParam.new("replace", "string", "Replacement text.")]
      ) do |args|
        path, err = AgentDemo.safe_path(args["path"]?)
        if path.nil?
          next err
        end
        next "Error: not a file: #{args["path"]?}" if Dir.exists?(path) || !File.exists?(path)
        find = args["find"]? || ""
        next "Error: empty 'find'" if find.empty?
        replace = args["replace"]? || ""
        content = File.read(path)
        count = content.scan(find).size
        next "No occurrences of the given text in #{args["path"]?}" if count == 0
        # Refuse a non-unique match instead of replacing every one of them.
        #
        # This used to gsub. A model asked to change one `end` rewrote every `end` in the
        # file, and the only hint was a count buried in the confirmation line, which an
        # approved-all session never shows. Ambiguity is the model's to resolve, not ours to
        # guess at, so hand it back and say how to fix it.
        if count > 1
          next "Error: 'find' matches #{count} times in #{args["path"]?}. Include more " \
               "surrounding context so it matches exactly once, or use apply_patch for a " \
               "multi-hunk change."
        end
        updated = content.sub(find, replace)
        STDERR.puts AgentDemo.preview_diff(path, updated) unless AgentDemo.allow_all?
        next "Declined by user." unless AgentDemo.confirm?("edit #{args["path"]?}")
        File.write(path, updated)
        "Edited #{args["path"]?} (replaced #{find.size} chars with #{replace.size} chars)"
      end,
      Tool.new(
        "apply_patch",
        <<-DESC,
          Apply a V4A context-diff patch to one or more files. PREFERRED for all file edits: it supports multi-file, multi-hunk changes in a single call, and edits are located by surrounding context rather than line numbers, so they survive minor drift. Read the file first so your context lines match.

          Format:
          *** Begin Patch
          *** Update File: path/to/file
          @@ optional anchor (e.g. a function name)
           unchanged context line
          -removed line
          +added line
          *** Add File: path/to/new-file
          +line 1
          *** Delete File: path/to/old-file
          *** End Patch

          Rules: ' ' prefix = context, '-' = remove, '+' = add. Include ~3 context lines around each change. Use @@ anchors when the context is not unique. All patch paths must be INSIDE the workspace: unlike read_file and edit_file, a patch cannot reach an approved outside location, so use edit_file for those.
          DESC
        [ToolParam.new("patch", "string", "Full patch in V4A format, including the *** Begin Patch / *** End Patch markers.")]
      ) do |args|
        patch = args["patch"]? || ""
        next "Error: empty patch" if patch.empty?
        # Show which files the patch touches BEFORE asking: the patch body is the diff, so a
        # separate preview would just repeat it, but the file list is what the approval is
        # actually about.
        targets = patch.lines.compact_map do |l|
          m = l.match(/^\*\*\* (Update|Add|Delete) File: (.+)$/)
          "#{m[1].downcase} #{m[2].strip}" if m
        end
        next "Error: no *** Update/Add/Delete File: directives found in the patch" if targets.empty?
        next "Declined by user." unless AgentDemo.confirm?("apply patch to #{targets.size} file(s): #{targets.join(", ")}")
        begin
          result = AgentDemo::V4A.apply(AgentDemo.workspace, patch)
        rescue ex : AgentDemo::V4A::PatchParseError
          next "Error: patch does not parse: #{ex.message}"
        rescue ex : AgentDemo::WorkspacePathError
          next "Error: #{ex.message}. Paths must be relative to the workspace."
        rescue ex
          next "Error applying patch: #{ex.message}"
        end
        lines = [] of String
        lines << "Applied: #{result[:applied].join(", ")}" unless result[:applied].empty?
        # Report partial success honestly: some hunks landing and others not is the normal
        # failure mode of context matching, and the model must know which is which to retry.
        lines << "Failed: #{result[:errors].join("; ")}" unless result[:errors].empty?
        lines << "Patch matched nothing." if lines.empty?
        lines.join("\n")
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
    SPINNER = %w[⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏]

    def initialize(@io : IO)
      @mode = :normal
      @pending = ""
      @started = false
      @tool_chars = 0
      @spin = 0
      @status_shown = false
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
      clear_status
      emit(@pending)
      @pending = ""
    end

    private def emit(t : String)
      return if t.empty?
      # Hidden tool-call tokens (the model writing a whole file inline) would
      # otherwise be dead silence — show a live spinner + byte count instead.
      if @mode == :tool
        @tool_chars += t.size
        @spin = (@spin + 1) % SPINNER.size
        @io.print "\r  #{"#{SPINNER[@spin]} writing tool call… (#{@tool_chars} chars)".colorize(:dark_gray)}"
        @io.flush
        @status_shown = true
        return
      end
      # Trim leading whitespace before the very first visible chars.
      unless @started
        t = t.lstrip
        return if t.empty?
        @started = true
      end
      case @mode
      when :think  then @io.print t.colorize(:dark_gray)
      when :normal then @io.print t
      end
      @io.flush
    end

    # Erase the in-progress status line (so following output starts clean).
    private def clear_status
      return unless @status_shown
      @io.print "\r" + (" " * 36) + "\r"
      @io.flush
      @status_shown = false
    end

    private def apply(action)
      case action
      when :think_open  then @mode = :think
      when :think_close then @mode = :normal
      when :tool_open   then @tool_chars = 0; @mode = :tool
      when :tool_close  then clear_status; @mode = :normal
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

    def initialize(@net : SHAInet::Network, @tokenizer : SHAInet::BPETokenizer, @tools : Array(Tool), @max_context : Int32 = 16384)
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

    # Same framing, but with the assistant's body taken VERBATIM from the tokens the model
    # emitted. This is what makes the KV prefix match: during generation the cache holds
    # `<|im_start|>assistant\n` followed by exactly these ids, so the cache is a true prefix
    # of the next prompt and only the tail has to be prefilled.
    private def render_assistant(body : Array(Int32)) : Array(Int32)
      ids = [@im_start]
      ids.concat(@tokenizer.encode("assistant\n"))
      ids.concat(body)
      ids << @im_end
      ids.concat(@nl)
      ids
    end

    private def build_prompt : Array(Int32)
      ids = render_message("system", @system_block)
      @messages.each do |m|
        if m.role == "tool"
          ids.concat(render_message("user", "<tool_response>\n#{m.content}\n</tool_response>"))
        elsif body = m.ids
          ids.concat(render_assistant(body))
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
      # Also drop the KV cache. Prefix reuse would stay correct without this -- a shorter
      # prompt cannot be an extension of the old cache, so it would fall back to a full
      # prefill -- but there is no reason to keep the old conversation's cache resident.
      reset_cache!
    end

    # Bound a tool result in TOKENS, not bytes.
    #
    # The tools already cap their output at MAX_READ_BYTES, but bytes are the wrong unit: the
    # binding constraint is the context window, and the bytes-per-token ratio varies wildly.
    # 64 KB of English is roughly 16k tokens; 64 KB of a compiled binary tokenized to 60534
    # in one observed run, which then tried to prefill against an 8192-token window.
    # Compaction cannot rescue that either, because it drops the OLDEST messages while the
    # offending one is the newest.
    #
    # A quarter of the window leaves room for the transcript, the primer and a reply.
    private def clamp_tool_result(result : String) : String
      budget = @max_context // 4
      ids = @tokenizer.encode(result)
      return result if ids.size <= budget
      kept = @tokenizer.decode(ids[0, budget]).scrub
      "#{kept}\n... [truncated: #{ids.size} tokens exceeded the #{budget}-token tool budget]"
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
      summary = generate_from(prompt, 384)[0].strip
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
        if cs[:hits] + cs[:misses] > 0
          parts << "cache #{(cs[:hit_rate] * 100).round}% hit"
        end
      end
      "[#{parts.join(" · ")}]"
    end

    # Tokens currently held in the KV cache, in order. This is what makes turn 2 onwards
    # cheap: the cache already contains the whole conversation so far, so only the new tail
    # has to be prefilled.
    @cache_ids = [] of Int32

    # Clearing the KV cache and forgetting what it held MUST happen together. If they drift,
    # a later turn believes the cache still holds a prefix that is gone and reuses it, which
    # is silently wrong output rather than an error.
    private def reset_cache!
      @net.clear_cache!
      @cache_ids.clear
    end

    # Prefill `prompt`, reusing whatever the KV cache already holds.
    #
    # This used to clear the cache and re-prefill the entire transcript on EVERY call, and
    # the tool loop calls it once per step -- so one request with three tool calls paid four
    # full prefills of a growing context. At 16k that is around 405 s each, which is the
    # multi-minute silence between an answer and the next action. It was never hidden
    # reasoning; it was prefill with nothing to show.
    #
    # build_prompt appends, so consecutive prompts share a long prefix. Reuse is only taken
    # when the cache is EXACTLY a prefix of the new prompt: a KV cache cannot drop a suffix,
    # so a partial match cannot be salvaged. If the assistant's text re-tokenizes differently
    # from the tokens that were generated, the common prefix simply ends early and this falls
    # back to a full prefill -- slower, never wrong.
    private def prefill(prompt : Array(Int32)) : SHAInet::SimpleMatrix
      shared = 0
      limit = Math.min(@cache_ids.size, prompt.size)
      while shared < limit && @cache_ids[shared] == prompt[shared]
        shared += 1
      end

      if shared == @cache_ids.size && shared > 0 && shared < prompt.size
        reused = shared
        tail = prompt[shared..]
        STDERR.puts "  reusing #{reused} cached tok · prefilling #{tail.size}".colorize(:dark_gray)
        logits = run_prefill(tail, reused, prompt.size)
        @cache_ids = prompt.dup
        logits
      else
        STDERR.puts "  prefilling #{prompt.size} tok from scratch".colorize(:dark_gray) if prompt.size > 512
        reset_cache!
        logits = run_prefill(prompt, 0, prompt.size)
        @cache_ids = prompt.dup
        logits
      end
    end

    # Feed tokens to the model in slices so there is something to show. The KV cache
    # accumulates across calls, so slicing is equivalent to one call.
    private def run_prefill(ids : Array(Int32), already : Int32, total : Int32) : SHAInet::SimpleMatrix
      slice = (ENV["SHAINET_PREFILL_SLICE"]? || "0").to_i
      slice = ids.size if slice <= 0
      logits = nil
      i = 0
      t0 = Time.monotonic
      while i < ids.size
        n = Math.min(slice, ids.size - i)
        logits = @net.run(ids[i, n], stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
        i += n
        done = already + i
        elapsed = (Time.monotonic - t0).total_seconds
        STDERR.print "\r  prefill #{done}/#{total} tok (#{done * 100 // total}%) · #{elapsed.round(0).to_i}s".colorize(:dark_gray)
        STDERR.flush
      end
      STDERR.print "\r\033[K"
      logits.not_nil!
    end

    # Generate from an explicit prompt. When echo is set, the user-facing prose is streamed
    # live (with <think> and tool-call markup filtered out).
    private def generate_from(prompt : Array(Int32), max_tokens : Int32, echo : Bool = false) : Tuple(String, Array(Int32))
      logits = prefill(prompt)
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
        # The sampled token is now in the cache too, so the next turn can reuse it.
        @cache_ids << id
      end
      renderer.try(&.finish)
      # scrub: a broken model can emit tokens that decode to invalid UTF-8, which
      # would crash downstream regex/parsing. The raw ids go back too: the decoded text is
      # for humans and parsing, the ids are what the KV cache actually holds.
      {@tokenizer.decode(generated).scrub, generated}
    end

    # Generate one assistant turn from the current transcript, streaming the
    # filtered prose live. Re-prefills the whole (growing) context each call.
    private def generate(max_tokens : Int32) : Tuple(String, Array(Int32))
      generate_from(build_prompt, max_tokens, echo: true)
    end

    # Handle one user input: run the tool loop until the model answers plainly.
    def chat(input : String, max_tokens : Int32)
      @messages << Message.new("user", input)
      step = 0
      loop do
        step += 1
        # The LAST allowed step still generates, it just may not call tools. Breaking before
        # generation (which is what this did) threw the turn away: the user got the cap
        # message and no answer, after the agent had already done the work. So tell the model
        # its budget is gone and let it write a closing reply from what it has.
        last_step = step >= MAX_TOOL_STEPS
        if step > MAX_TOOL_STEPS
          break
        end
        if last_step
          STDERR.puts "  [agent] tool budget reached (#{MAX_TOOL_STEPS} steps); asking for a final answer".colorize(:dark_gray)
          STDERR.puts "  (raise it with AGENT_MAX_TOOL_STEPS)".colorize(:dark_gray)
          @messages << Message.new("user",
            "You have used all #{MAX_TOOL_STEPS} tool steps for this turn. Do not call any " \
            "more tools. Answer now with what you already have, and say plainly what you " \
            "could not finish.")
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
          text, text_ids = generate(max_tokens) # streams the filtered prose inline
        rescue ex
          STDERR.puts "\n  [agent] generation failed: #{ex.message}".colorize(:red)
          STDERR.puts "  (out of GPU memory? try /clear, a shorter request, or a smaller SHAINET_EXPERT_CACHE_MB)".colorize(:dark_gray)
          reset_cache!
          break
        end
        STDERR.puts ""
        calls = AgentDemo.parse_tool_calls(text)

        # If the model generated only think tags or whitespace, the user sees nothing.
        visible = text.gsub(/<think>.*?<\/think>/m, "").gsub(/<tool_call>.*?<\/tool_call>/m, "").strip
        if visible.empty? && calls.empty?
          STDERR.puts "  [agent] model returned no visible text".colorize(:dark_gray)
        end

        # Keep the assistant's output (incl. any tool_call markup) verbatim.
        @messages << Message.new("assistant", text.strip, text_ids)
        break if calls.empty?
        # On the final step the reply IS the answer. Running its tool calls would spend the
        # results and then break on the next iteration, throwing them away unanswered.
        break if last_step

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
          @messages << Message.new("tool", clamp_tool_result(result))
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
# Per-response generation ceiling. This is a safety bound, not a target — the
# model stops itself on <|im_end|>. It must be large enough that a tool call
# writing a whole file isn't truncated before its closing </tool_call> tag
# (which would make the call unparseable). KV cache grows ~linearly with it.
max_tokens = (ARGV[1]? || ENV["SHAINET_AGENT_MAX_TOKENS"]? || "4096").to_i
unless model_dir && Dir.exists?(model_dir)
  STDERR.puts "Usage: agent <model-dir> [max_tokens]"
  STDERR.puts "  (point at an already-downloaded model, e.g. ~/models/Qwen3-Coder-30B-A3B-Instruct)"
  exit 1
end

Colorize.enabled = STDERR.tty?
STDERR.sync = true # stream tokens as they arrive (no buffering)
AgentDemo.print_banner(STDERR, "#{File.basename(model_dir)} · local coding agent on Network#run")

STDERR.puts "Loading model from #{model_dir}...".colorize(:dark_gray)
# A 30B takes about nine minutes here. Without a progress line that is indistinguishable
# from a hang, and the layer loop is where nearly all of it goes.
load_started = Time.monotonic
SHAInet::HFLoader.progress = ->(done : Int32, total : Int32) do
  elapsed = (Time.monotonic - load_started).total_seconds
  eta = done > 0 ? (elapsed / done) * (total - done) : 0.0
  STDERR.print "\r  layer #{done}/#{total} (#{done * 100 // total}%) " \
               "· #{elapsed.round.to_i}s elapsed · ~#{eta.round.to_i}s left".colorize(:dark_gray)
  STDERR.print "\n" if done == total
end
t0 = Time.monotonic
# Q4 + MoE offload are the only configuration a large MoE model actually runs in on a
# single consumer GPU: without them a 30B-A3B does not fit at all. So they are the
# DEFAULT here rather than opt-in. SHAINET_Q8=1 and SHAINET_MOE_OFFLOAD=0 opt out.
ENV["SHAINET_MOE_OFFLOAD"] = "1" unless ENV.has_key?("SHAINET_MOE_OFFLOAD")
quantize = SHAInet::CUDA.fully_available? && !ENV["SHAINET_FP32"]?
bits = ENV.fetch("SHAINET_Q8", "0") == "1" ? 8 : 4
offload = ENV.fetch("SHAINET_MOE_OFFLOAD", "0") == "1"
net = SHAInet::HFLoader.load(model_dir, quantize: quantize, bits: bits)
net.use_kv_cache = true
tokenizer = SHAInet::BPETokenizer.from_hf(File.join(model_dir, "tokenizer.json"))
has_moe = net.hidden_layers.any? { |l| l.is_a?(SHAInet::LlamaBlock) && l.as(SHAInet::LlamaBlock).ffn.is_a?(SHAInet::MoEFF) }
mode = ENV["SHAINET_FP32"]? ? "fp32" : "Q#{bits}"
mode += " (MoE offload)" if offload && has_moe
STDERR.puts "Loaded in #{(Time.monotonic - t0).total_seconds.round(1)}s · #{mode} (vocab #{tokenizer.vocab.size})"

# Size the expert cache to leave headroom for the model + prefill activations.
# Only relevant for MoE models, where a fraction of experts are active per token and the rest
# are cached in VRAM for reuse. Dense models (Qwen3.5, Qwen3-0.6B) have no experts to cache,
# so reserving VRAM for this wastes 2+ GB that could serve KV context instead.
# Override with SHAINET_EXPERT_CACHE_MB (0 disables).
max_context = (ENV["SHAINET_AGENT_CONTEXT"]? || "16384").to_i

has_moe = net.hidden_layers.any? { |l| l.is_a?(SHAInet::LlamaBlock) && l.as(SHAInet::LlamaBlock).ffn.is_a?(SHAInet::MoEFF) }

# The reserve has to cover what GROWS with context -- the KV cache above all -- not just a
# flat allowance. Measured on Qwen3-Coder-30B-A3B: with a fixed 6 GB reserve the cache took
# 7881 MB and a 16384-token prefill peaked at 15140 MB of 16376, leaving 1.2 GB. That is
# where a multi-turn agent runs out. Capping the cache instead put the same prefill at
# 11455 MB, and 24576 tokens fit in 11069 MB.
#
# 128 KiB per token is deliberately generous: fp16 KV on this 48-layer 30B is about 96
# KiB/token, and the slack covers the prefill workspaces. A smaller model over-reserves,
# which costs it cache it did not need anyway since it is not near the VRAM limit.
# SHAINET_EXPERT_CACHE_MB overrides the whole calculation (0 disables the cache).
if has_moe && offload && !ENV["SHAINET_EXPERT_CACHE_MB"]? && (info = SHAInet::CUDA.memory_info)
  kv_reserve = max_context.to_u64 * 128_u64 * 1024_u64
  reserve = 6_u64 * 1024 * 1024 * 1024 + kv_reserve
  free = info[:free]
  budget_mb = free > reserve ? ((free - reserve) // (1024_u64 * 1024_u64)) : 0_u64
  ENV["SHAINET_EXPERT_CACHE_MB"] = budget_mb.to_s
  STDERR.puts "  Expert cache budget: #{budget_mb} MB " \
              "(free #{free // (1024*1024)} MB − 6GB − #{kv_reserve // (1024*1024)} MB for #{max_context} tok context)".colorize(:dark_gray)
end

agent = AgentDemo::Agent.new(net, tokenizer, AgentDemo.build_tools, max_context)
STDERR.puts "Ready · tools: #{AgentDemo.build_tools.map(&.name).join(", ")} · max context #{max_context} tok".colorize(:green)
STDERR.puts "Commands: /context  /compact  /clear  /ask  /help   (Ctrl-D to exit)".colorize(:dark_gray)

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
    STDERR.puts "  /ask      revoke 'allow all' and confirm each tool call again".colorize(:dark_gray)
    STDERR.puts "  /help     this message".colorize(:dark_gray)
    next
  when "/ask"
    # The way back from answering "a". Without this, one keystroke silently approves every
    # write and shell command for the rest of the session with no way to reconsider. It also
    # forgets approved out-of-workspace locations, since those were granted by the same
    # keystroke and leaving them behind would make the revocation only partly true.
    outside = AgentDemo::WorkspacePath.approved_outside_count
    if AgentDemo.allow_all? || outside > 0
      AgentDemo.allow_all = false
      AgentDemo::WorkspacePath.reset_outside_approvals!
      note = outside > 0 ? " (also forgot #{outside} approved path(s) outside the workspace)" : ""
      STDERR.puts "  will confirm each tool call again#{note}".colorize(:dark_gray)
    else
      STDERR.puts "  already confirming each tool call".colorize(:dark_gray)
    end
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
SHAInet::Profile.report(STDERR)
