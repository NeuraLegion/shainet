require "../src/shainet"
require "./tool_protocol"
require "http/server"
require "json"
require "random/secure"

# OpenAI-compatible HTTP API server backed by SHAInet inference (Network#run).
#
# Exposes a subset of the OpenAI REST API so existing OpenAI client libraries
# (python `openai`, `curl`, LangChain, etc.) can talk to a locally-loaded model:
#
#   GET  /v1/models               — list the loaded model
#   POST /v1/chat/completions      — chat completion (streaming + non-streaming)
#   GET  /health                   — liveness probe
#
# Usage:
#   crystal run examples/openai_server.cr -Denable_cuda -- /path/to/model-dir [port] [host]
#   SHAINET_Q4=1 SHAINET_MOE_OFFLOAD=1 crystal run examples/openai_server.cr -Denable_cuda -- /models/Qwen3-Coder-30B-A3B-Instruct
#
# Quantization / offload are controlled by the same env vars as llama_chat.cr
# (SHAINET_FP32, SHAINET_Q4, SHAINET_MOE_OFFLOAD, SHAINET_EXPERT_CACHE_MB).
#
# SECURITY: this is an example server with NO authentication by default and it
# binds to 127.0.0.1 (localhost only). Do not expose it to an untrusted network.
# Set SHAINET_API_KEY to require an `Authorization: Bearer <key>` header, and
# pass an explicit host (e.g. 0.0.0.0) only when you understand the exposure.
module OpenAIServer
  # Logged under its own source so server lines are distinguishable from the library's `sha_inet:` ones
  # at a glance, and can be filtered separately.
  Log = ::Log.for("api")
  extend self

  # ---- OpenAI request schema (the fields we honor; unknown fields ignored) ----

  # One tool call, in the shape a client sends back and we emit.
  struct ToolCallRef
    include JSON::Serializable
    getter id : String
    getter type : String = "function"
    getter function : FunctionRef
  end

  struct FunctionRef
    include JSON::Serializable
    getter name : String
    getter arguments : String = "{}"
  end

  struct ChatMessage
    include JSON::Serializable
    getter role : String
    # Nilable because an assistant message that ONLY calls tools has content: null, and clients echo
    # that message straight back into the next request. A non-nilable field here rejected the whole
    # conversation with a 400 on the second turn of every tool-using session.
    getter content : String?
    getter tool_calls : Array(ToolCallRef)?
    getter tool_call_id : String?
  end

  struct ToolDef
    include JSON::Serializable
    getter type : String = "function"
    getter function : ToolFunctionDef
  end

  struct ToolFunctionDef
    include JSON::Serializable
    getter name : String
    getter description : String = ""
    getter parameters : JSON::Any?
  end

  struct ChatCompletionRequest
    include JSON::Serializable
    getter model : String?
    getter messages : Array(ChatMessage)
    getter? stream : Bool = false
    getter temperature : Float64?
    getter top_p : Float64?
    getter max_tokens : Int32?
    getter seed : Int64?
    getter frequency_penalty : Float64?
    getter presence_penalty : Float64?
    # `stop` may be a single string or an array of strings in the OpenAI API.
    getter stop : JSON::Any?
    getter tools : Array(ToolDef)?
    # "auto", "none", "required", or {type: "function", function: {name: ...}}.
    getter tool_choice : JSON::Any?

    # The function name a client is forcing, if any.
    def forced_tool : String?
      tc = @tool_choice
      return unless tc
      tc["function"]?.try(&.["name"]?).try(&.as_s?)
    end

    def tool_defs : Array(ToolProtocol::FunctionDef)
      (@tools || [] of ToolDef).map do |t|
        ToolProtocol::FunctionDef.new(t.function.name, t.function.description, t.function.parameters)
      end
    end
  end

  # Result of one generation pass.
  record Generation, text : String, prompt_tokens : Int32,
    completion_tokens : Int32, finish_reason : String,
    calls : Array(ToolProtocol::Call) = [] of ToolProtocol::Call,
    tool_defs : Array(ToolProtocol::FunctionDef) = [] of ToolProtocol::FunctionDef,
    reasoning : String = ""

  # Wraps a loaded model + tokenizer and serializes generation. The Network's
  # KV cache is shared mutable state, so only one request may generate at a
  # time — concurrent requests are queued behind @mutex.
  # Raised when a request cannot fit the model's context. Carried as its own type so the handler can
  # answer 400 with a message a client can act on, rather than letting an allocation failure take the
  # process down.
  class ContextLengthError < Exception
  end

  class Engine
    getter model_name : String

    @mutex = Mutex.new
    @nl : Array(Int32)
    @nl2 : Array(Int32)
    @bos : Int32?
    @start_hdr : Int32?
    @end_hdr : Int32?
    @eot : Int32?
    @im_start : Int32?
    @im_end : Int32?
    @stop_ids : Array(Int32)

    def initialize(@net : SHAInet::Network, @tok : SHAInet::BPETokenizer, @model_name : String,
                   @max_context : Int32 = 65536)
      @net.use_kv_cache = true
      # Special tokens used by the supported chat templates (nil when absent).
      @bos = sp("<|begin_of_text|>")
      @start_hdr = sp("<|start_header_id|>")
      @end_hdr = sp("<|end_header_id|>")
      @eot = sp("<|eot_id|>")
      @im_start = sp("<|im_start|>")
      @im_end = sp("<|im_end|>")
      @nl = @tok.encode("\n")
      @nl2 = @tok.encode("\n\n")
      # End-of-turn / end-of-text ids that always terminate generation.
      @stop_ids = [] of Int32
      ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"].each do |name|
        if id = sp(name)
          @stop_ids << id unless @stop_ids.includes?(id)
        end
      end
    end

    private def sp(name : String) : Int32?
      @tok.vocab[name]?
    end

    # Flatten the OpenAI message list into {role, content} pairs the chat template can carry.
    #
    # Three shapes need translating, and the model only understands the last of them:
    #
    #   * the tool DECLARATIONS, which OpenAI passes as a separate `tools` array and this model expects
    #     as XML inside the system message. When the client sent no system message one is created, or
    #     the block would have nowhere to live.
    #   * an ASSISTANT message carrying tool_calls and usually content: null -- the client echoes its
    #     own previous turn back. Re-rendered as the XML the model originally emitted, so the
    #     conversation it sees is the one it wrote rather than a gap where its call used to be.
    #   * a TOOL result, which OpenAI sends as its own role with a tool_call_id. This model was trained
    #     on <tool_response> inside a USER turn, so that is what it gets.
    private def render_messages(messages : Array(ChatMessage),
                                tools : Array(ToolProtocol::FunctionDef),
                                forced_tool : String?) : Array(Tuple(String, String))
      result = [] of Tuple(String, String)
      tool_block = tools.empty? ? "" : ToolProtocol.render_tools(tools)
      tool_block += ToolProtocol.render_forced_choice(forced_tool) if forced_tool && !tools.empty?
      injected = tool_block.empty?

      messages.each do |m|
        case m.role
        when "system"
          body = m.content || ""
          unless injected
            body = "#{body}\n\n#{tool_block}"
            injected = true
          end
          result << {"system", body}
        when "tool"
          result << {"user", "<tool_response>\n#{m.content || ""}\n</tool_response>"}
        when "assistant"
          body = m.content || ""
          if calls = m.tool_calls
            calls.each do |c|
              body += "\n" unless body.empty?
              body += render_call_xml(c)
            end
          end
          result << {"assistant", body}
        else
          result << {m.role, m.content || ""}
        end
      end

      # No system message to attach the declarations to: prepend one rather than dropping the tools.
      result.unshift({"system", tool_block}) unless injected
      result
    end

    # Re-render a client-supplied tool call as the XML the model emits, so its own prior turn reads
    # back to it in the format it produced. Arguments arrive as a JSON string and are unpacked to one
    # <parameter=> each; a value that is not a plain scalar is written back as JSON, which is what the
    # model was asked to produce for an array or object parameter.
    private def render_call_xml(c : ToolCallRef) : String
      params = String.build do |s|
        JSON.parse(c.function.arguments).as_h.each do |k, v|
          s << "<parameter=" << k << ">\n" << (v.as_s? || v.to_json) << "\n</parameter>\n"
        end
      rescue
        # Unparseable arguments: keep them verbatim rather than silently emitting an empty call.
        s << "<parameter=arguments>\n" << c.function.arguments << "\n</parameter>\n"
      end
      "<tool_call>\n<function=#{c.function.name}>\n#{params}</function>\n</tool_call>"
    end

    # Render an OpenAI message list into prompt token ids using whichever chat
    # template the tokenizer advertises (ChatML, then LLaMA 3, else a plain
    # concatenation), ending with the assistant generation prompt.
    def build_prompt(messages : Array(ChatMessage),
                     tools : Array(ToolProtocol::FunctionDef) = [] of ToolProtocol::FunctionDef,
                     forced_tool : String? = nil) : Array(Int32)
      ids = [] of Int32
      im_start, im_end = @im_start, @im_end
      bos, start_hdr, end_hdr, eot = @bos, @start_hdr, @end_hdr, @eot
      rendered = render_messages(messages, tools, forced_tool)

      if im_start && im_end
        rendered.each do |role, content|
          ids << im_start
          ids.concat(@tok.encode(role))
          ids.concat(@nl)
          ids.concat(@tok.encode(content))
          ids << im_end
          ids.concat(@nl)
        end
        ids << im_start
        ids.concat(@tok.encode("assistant"))
        ids.concat(@nl)
      elsif bos && start_hdr && end_hdr && eot
        ids << bos
        rendered.each do |role, content|
          ids << start_hdr
          ids.concat(@tok.encode(role))
          ids << end_hdr
          ids.concat(@nl2)
          ids.concat(@tok.encode(content))
          ids << eot
        end
        ids << start_hdr
        ids.concat(@tok.encode("assistant"))
        ids << end_hdr
        ids.concat(@nl2)
      else
        ids << (bos || 0)
        messages.each do |m|
          ids.concat(@tok.encode("#{m.role}: #{m.content}"))
          ids.concat(@nl)
        end
      end
      ids
    end

    # Run one generation. Each newly-decoded text piece is yielded to the block
    # (used for SSE streaming); the full text is also returned. Generation is
    # serialized across requests via @mutex.
    # Pre-flight the context budget, returning an error message or nil.
    #
    # Called BEFORE the handler decides between streaming and not, because a streaming response has
    # already sent its 200 and its SSE headers by the time generation starts -- at that point a 400 is
    # no longer possible and the client would get a stream that simply dies. Re-tokenizing the prompt
    # to answer this costs a fraction of one token's generation.
    def context_error(messages : Array(ChatMessage), max_tokens : Int32) : String?
      needed = build_prompt(messages).size
      return if needed + max_tokens <= @max_context
      "This model's maximum context length is #{@max_context} tokens, but the request needs " \
      "#{needed} prompt tokens + #{max_tokens} completion tokens (#{needed + max_tokens}). " \
      "Shorten the messages or lower max_tokens."
    end

    def generate(messages : Array(ChatMessage), max_tokens : Int32,
                 temperature : Float64, top_k : Int32, repetition_penalty : Float64,
                 stop_strings : Array(String), seed : Int64?,
                 tools : Array(ToolProtocol::FunctionDef) = [] of ToolProtocol::FunctionDef,
                 forced_tool : String? = nil, &block : String ->) : Generation
      @mutex.synchronize do
        @net.clear_cache!
        prompt_ids = build_prompt(messages, tools, forced_tool)
        # Refuse a prompt that cannot fit rather than dying on it.
        #
        # The agent controls its own prompt and compacts when it grows; a server is handed whatever a
        # client sends. Without this, one oversized request exhausts VRAM and takes the process down,
        # losing every other client's session with it. OpenAI's own API answers this case with an
        # error, so a client already knows how to handle it.
        if prompt_ids.size + max_tokens > @max_context
          raise ContextLengthError.new(
            "This model's maximum context length is #{@max_context} tokens, but the request needs " \
            "#{prompt_ids.size} prompt tokens + #{max_tokens} completion tokens " \
            "(#{prompt_ids.size + max_tokens}). Shorten the messages or lower max_tokens.")
        end
        rng = seed ? Random.new(seed) : Random.new
        sampler = SHAInet::Sampler.new(temperature: temperature, top_k: top_k,
          repetition_penalty: repetition_penalty, rng: rng)

        # Prefill is timed and logged separately from decode because they fail differently and are
        # bound by different things. A slow prefill means the prompt is large or the cache missed; slow
        # decode means layers did not fit on the card. One combined duration cannot tell an operator
        # which of those is happening, and on this model a host-resident layer costs about a second per
        # TOKEN, so the difference is the whole diagnosis.
        pf = Time.instant
        logits = @net.run(prompt_ids, stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
        pf_secs = (Time.instant - pf).total_seconds
        Log.info do
          "prefill #{prompt_ids.size} tok in #{pf_secs.round(2)}s " \
          "(#{(prompt_ids.size / (pf_secs > 0 ? pf_secs : 1.0)).round(0).to_i} tok/s)"
        end
        dec = Time.instant
        generated = [] of Int32
        text = ""
        emitted = 0
        finish = "length"

        max_tokens.times do
          # Let other fibers run.
          #
          # Crystal's HTTP server is fiber-based, and this loop is CPU/GPU bound with no yield point of
          # its own. Without this the WHOLE server is frozen for the duration of a generation -- not
          # just this request: the accept loop stops, health checks time out, and a streaming response
          # cannot flush. At roughly 85 ms per token a 500-token answer is 42 seconds of a server that
          # looks dead to everything else. A streaming request gets away with it by accident, because
          # writing each chunk is I/O and yields; a non-streaming one does no I/O per token at all.
          #
          # Same root cause as the agent's Ctrl-C handler never firing: a tight Crystal loop starves
          # every other fiber, including the ones the runtime needs.
          Fiber.yield
          last = logits.rows - 1
          sampler.apply_repetition_penalty!(logits, generated, window: 64, row: last)
          id = sampler.sample(logits, last)
          break if id < 0 || @stop_ids.includes?(id) ? (finish = "stop"; true) : false
          break unless logits[last, id].finite?

          generated << id
          text += @tok.decode([id])

          # Honor user-supplied stop sequences: cut at the first occurrence.
          if si = first_stop_index(text, stop_strings)
            piece = text[emitted...si]
            block.call(piece) unless piece.empty?
            text = text[0...si]
            finish = "stop"
            break
          end

          block.call(text[emitted..])
          emitted = text.size

          logits = @net.run([id], stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
        end

        # Extract any tool calls the model emitted, and report the finish reason OpenAI clients branch
        # on. A client decides whether to run tools by the presence of tool_calls, so a call the parser
        # misses is indistinguishable from a final answer -- which is exactly how a truncated call once
        # got printed as one.
        calls = tools.empty? ? [] of ToolProtocol::Call : ToolProtocol.parse_calls(text)
        finish = "tool_calls" unless calls.empty?
        reasoning, answer = OpenAIServer.split_reasoning(text)
        dec_secs = (Time.instant - dec).total_seconds
        ms_per_tok = generated.size > 0 ? (dec_secs * 1000 / generated.size).round(0).to_i : 0
        Log.info do
          "decode #{generated.size} tok in #{dec_secs.round(2)}s (#{ms_per_tok} ms/tok) " \
          "finish=#{finish}#{calls.empty? ? "" : " calls=#{calls.map(&.name).join(",")}"}" \
          "#{reasoning.empty? ? "" : " reasoning=#{reasoning.size}ch"}"
        end
        Generation.new(answer, prompt_ids.size, generated.size, finish, calls, tools, reasoning)
      end
    end

    private def first_stop_index(text : String, stops : Array(String)) : Int32?
      idx = nil
      stops.each do |s|
        next if s.empty?
        if i = text.index(s)
          idx = idx ? Math.min(idx, i) : i
        end
      end
      idx
    end
  end

  # ---- response helpers -------------------------------------------------------

  def completion_id : String
    "chatcmpl-#{Random::Secure.hex(12)}"
  end

  # Separate the model's private reasoning from its answer. Returns {reasoning, answer}.
  #
  # Every OpenAI-compatible server that handles reasoning models agrees on the invariant: `content`
  # carries the conclusion only. They differ on the reasoning, and none of them throw it away -- DeepSeek
  # returns it as `reasoning_content` "at the same level as content", vLLM the same and now migrating to
  # `reasoning`, Anthropic as a separate thinking block, and OpenAI's own reasoning models withhold the
  # text but still bill it in usage. An earlier version of this code deleted it, which kept content
  # correct and made the server useless to a client that wants to display or log the chain of thought.
  #
  # An unterminated block -- generation stopped mid-thought -- yields all reasoning and no answer, which
  # is what actually happened: there is no conclusion to report, and inventing one from the scratchpad
  # would be worse than an empty answer.
  def split_reasoning(text : String) : Tuple(String, String)
    reasoning = [] of String
    text.scan(/<think>(.*?)<\/think>/m) { |m| reasoning << m[1].strip }
    answer = text.gsub(/<think>.*?<\/think>/m, "")
    if i = answer.index("<think>")
      reasoning << answer[(i + "<think>".size)..].strip
      answer = answer[0, i]
    end
    {reasoning.reject(&.empty?).join("\n\n"), answer.strip}
  end

  def chat_completion_json(id : String, model : String, gen : Generation) : String
    JSON.build do |j|
      j.object do
        j.field "id", id
        j.field "object", "chat.completion"
        j.field "created", Time.utc.to_unix
        j.field "model", model
        j.field "choices" do
          j.array do
            j.object do
              j.field "index", 0
              j.field "message" do
                j.object do
                  j.field "role", "assistant"
                  # Both names, because the ecosystem is mid-rename: DeepSeek and older vLLM read
                  # `reasoning_content`, current vLLM reads `reasoning` and warns that a client still
                  # reading the old name will silently see nothing. Emitting one would quietly fail for
                  # half of them, and an unknown field is ignored by every client, so emitting both is
                  # the only option with no silent-failure mode. Omitted entirely when there was no
                  # reasoning, so a non-thinking turn does not carry empty fields.
                  unless gen.reasoning.empty?
                    j.field "reasoning_content", gen.reasoning
                    j.field "reasoning", gen.reasoning
                  end
                  # content is null when the turn is only a tool call, which is what OpenAI does and
                  # what a client's own type expects. The XML is stripped either way: a client that
                  # received the raw <tool_call> markup as content would show it to its user and, worse,
                  # echo it back as text on the next turn alongside the structured call.
                  if gen.calls.empty?
                    j.field "content", gen.text
                  else
                    j.field "content", nil
                    j.field "tool_calls" do
                      j.array do
                        gen.calls.each do |c|
                          definition = gen.tool_defs.find { |d| d.name == c.name }
                          j.object do
                            j.field "id", c.id
                            j.field "type", "function"
                            j.field "function" do
                              j.object do
                                j.field "name", c.name
                                # A JSON *string*, not an object: the client calls JSON.parse on it.
                                j.field "arguments", ToolProtocol.arguments_json(c, definition)
                              end
                            end
                          end
                        end
                      end
                    end
                  end
                end
              end
              j.field "finish_reason", gen.finish_reason
            end
          end
        end
        j.field "usage" do
          j.object do
            j.field "prompt_tokens", gen.prompt_tokens
            j.field "completion_tokens", gen.completion_tokens
            j.field "total_tokens", gen.prompt_tokens + gen.completion_tokens
          end
        end
      end
    end
  end

  # One SSE chat.completion.chunk. `delta_role`/`delta_content` are optional;
  # `finish_reason` is nil for content chunks and set on the final chunk.
  def chunk_json(id : String, model : String, created : Int64,
                 delta_role : String? = nil, delta_content : String? = nil,
                 finish_reason : String? = nil) : String
    JSON.build do |j|
      j.object do
        j.field "id", id
        j.field "object", "chat.completion.chunk"
        j.field "created", created
        j.field "model", model
        j.field "choices" do
          j.array do
            j.object do
              j.field "index", 0
              j.field "delta" do
                j.object do
                  j.field "role", delta_role if delta_role
                  j.field "content", delta_content if delta_content
                end
              end
              j.field "finish_reason", finish_reason
            end
          end
        end
      end
    end
  end

  def error_json(message : String, type : String = "invalid_request_error") : String
    JSON.build do |j|
      j.object do
        j.field "error" do
          j.object do
            j.field "message", message
            j.field "type", type
          end
        end
      end
    end
  end

  def normalize_stop(stop : JSON::Any?) : Array(String)
    return [] of String unless stop
    if s = stop.as_s?
      [s]
    elsif arr = stop.as_a?
      arr.compact_map(&.as_s?)
    else
      [] of String
    end
  end
end

# =============================================================================
# Entry point
# =============================================================================

model_dir = ARGV[0]?
port = (ARGV[1]? || "8080").to_i
host = ARGV[2]? || "127.0.0.1"

# Accept the same three model forms the agent does: a SafeTensors directory, a .gguf FILE, or an Ollama
# model name.
#
# The server only ever accepted a directory, so pointing it at the .gguf everything else in this repo
# runs was rejected by the usage check -- and passing the containing directory instead got further and
# then failed looking for config.json, which reads like a corrupt download rather than a server that
# does not support the format. Same resolution, same tokenizer split, so the two entry points cannot
# drift on which models they accept.
gguf_mode = false
if model_dir && SHAInet::OllamaResolve.ollama_name?(model_dir)
  resolved = SHAInet::OllamaResolve.resolve(model_dir)
  unless resolved
    STDERR.puts "Ollama model '#{model_dir}' not found. Is it pulled?"
    exit 1
  end
  STDERR.puts "Resolved Ollama '#{model_dir}' -> #{resolved}"
  model_dir = resolved
  gguf_mode = true
elsif model_dir && File.file?(model_dir)
  gguf_mode = true
end

unless model_dir && (Dir.exists?(model_dir) || File.file?(model_dir))
  STDERR.puts "Usage: openai_server <model-dir | gguf-file | ollama-name> [port] [host]"
  STDERR.puts "  openai_server ~/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf 8080"
  STDERR.puts "  openai_server qwen3.8:27b"
  STDERR.puts "  openai_server ~/models/Qwen3-0.6B        # SafeTensors directory"
  exit 1
end

# --- Load model (same knobs as llama_chat.cr) ---
STDERR.puts "Loading model from #{model_dir}..."
t = Time.instant
quantize = SHAInet::CUDA.fully_available? && !ENV["SHAINET_FP32"]?
bits = ENV["SHAINET_Q4"]? ? 4 : 8
mode = ENV["SHAINET_FP32"]? ? "fp32" : "Q#{bits}"
offload = ENV.fetch("SHAINET_MOE_OFFLOAD", "0") == "1"
STDERR.puts "  Mode: #{mode}#{offload ? " (MoE offload)" : ""}"

# Decide the context BEFORE loading, and tell the loader.
#
# The loader sizes its KV reserve from SHAINET_MAX_CONTEXT at load time, so a context decided after
# the model is in memory is a context the loader never reserved for. The agent had exactly this bug:
# two independent defaults that happened to agree, and when they disagreed the symptom was an OOM
# part-way through a conversation rather than an error at startup. A server is worse off than the
# agent here, because it will accept whatever prompt a client sends.
#
# 65536 by default: this is what the GSQ-RCO model's own authors run (--ctx-size 65536), and on a
# 16 GB card it still places all 64 layers.
server_context = (ENV["SHAINET_SERVER_CONTEXT"]? || ENV["SHAINET_MAX_CONTEXT"]? || "65536").to_i
ENV["SHAINET_MAX_CONTEXT"] = server_context.to_s
STDERR.puts "  Context: #{server_context} tok (KV reserved for this at load)"

net = SHAInet::HFLoader.load(model_dir, quantize: quantize, bits: bits)
# A GGUF carries its own tokenizer; there is no tokenizer.json beside it. Reading one unconditionally
# is what turned "this format is not supported here" into a missing-file error about the wrong file.
tokenizer =
  if gguf_mode
    SHAInet::GGUF.extract_tokenizer(model_dir)
  else
    path = ENV["SHAINET_TOKENIZER_PATH"]? || File.join(model_dir, "tokenizer.json")
    unless File.exists?(path)
      STDERR.puts "Error: tokenizer.json not found at #{path} (set SHAINET_TOKENIZER_PATH to override)."
      exit 1
    end
    SHAInet::BPETokenizer.from_hf(path)
  end
STDERR.puts "Loaded in #{(Time.instant - t).total_seconds.round(1)}s (vocab #{tokenizer.vocab.size})"

# Clients display and echo this, so drop the .gguf extension rather than advertising a filename.
model_name = File.basename(model_dir.rstrip("/")).sub(/\.gguf$/i, "")
engine = OpenAIServer::Engine.new(net, tokenizer, model_name, server_context)

api_key = ENV["SHAINET_API_KEY"]?
default_max_tokens = (ENV["SHAINET_MAX_TOKENS"]? || "512").to_i

authorized = ->(req : HTTP::Request) : Bool {
  return true unless k = api_key
  req.headers["Authorization"]? == "Bearer #{k}"
}

server = HTTP::Server.new do |ctx|
  req = ctx.request
  res = ctx.response
  started = Time.instant
  peer = req.remote_address.try(&.to_s) || "-"

  begin
    unless authorized.call(req)
      res.status = HTTP::Status::UNAUTHORIZED
      res.content_type = "application/json"
      res.print OpenAIServer.error_json("Missing or invalid API key.", "invalid_request_error")
      next
    end

    # Accept routes with or without the `/v1` prefix: some OpenAI clients are
    # configured with a base URL that already includes `/v1`, others without.
    route = req.path.sub(/^\/v1/, "")

    case {req.method, route}
    when {"GET", "/health"}
      res.content_type = "application/json"
      res.print %({"status":"ok"})
    when {"GET", "/models"}
      res.content_type = "application/json"
      models = JSON.build do |j|
        j.object do
          j.field "object", "list"
          j.field "data" do
            j.array do
              j.object do
                j.field "id", model_name
                j.field "object", "model"
                j.field "created", Time.utc.to_unix
                j.field "owned_by", "shainet"
              end
            end
          end
        end
      end
      res.print models
    when {"POST", "/chat/completions"}
      body = req.body.try(&.gets_to_end) || ""
      request =
        begin
          OpenAIServer::ChatCompletionRequest.from_json(body)
        rescue ex : JSON::ParseException | ArgumentError
          res.status = HTTP::Status::BAD_REQUEST
          res.content_type = "application/json"
          res.print OpenAIServer.error_json("Invalid request body: #{ex.message}")
          next
        end

      if request.messages.empty?
        res.status = HTTP::Status::BAD_REQUEST
        res.content_type = "application/json"
        res.print OpenAIServer.error_json("'messages' must not be empty.")
        next
      end

      max_tokens = request.max_tokens || default_max_tokens
      temperature = request.temperature || 0.7
      temperature = 0.01 if temperature <= 0.0 # sampler requires > 0; ~greedy
      # OpenAI exposes top_p, not top_k; approximate with a fixed top_k and map
      # frequency/presence penalty onto SHAInet's repetition penalty.
      top_k = 40
      rep_pen = 1.0 + (request.frequency_penalty || request.presence_penalty || 0.0).clamp(0.0, 1.0)
      stops = OpenAIServer.normalize_stop(request.stop)
      model = request.model || model_name

      # What the caller actually asked for. Logged before generation rather than after, so a request that
      # hangs or dies mid-way still shows what it was -- which is the case where you most want to know.
      # Message ROLES rather than content: the shape is what explains a bad answer (a missing system
      # message, a tool result that never arrived), and prompts are the caller's data, not ours to spill
      # into a log file.
      OpenAIServer::Log.info do
        roles = request.messages.map(&.role).join(">")
        tool_names = request.tool_defs.map(&.name)
        "#{peer} chat model=#{model} msgs=#{request.messages.size} [#{roles}] " \
        "max_tokens=#{max_tokens} temp=#{temperature.round(2)}#{request.stream? ? " stream" : ""}" \
        "#{tool_names.empty? ? "" : " tools=#{tool_names.join(",")}"}" \
        "#{request.forced_tool ? " forced=#{request.forced_tool}" : ""}" \
        "#{stops.empty? ? "" : " stops=#{stops.size}"}"
      end

      if msg = engine.context_error(request.messages, max_tokens)
        res.status = HTTP::Status::BAD_REQUEST
        res.content_type = "application/json"
        res.print OpenAIServer.error_json(msg, "context_length_exceeded")
        next
      end

      if request.stream?
        res.content_type = "text/event-stream"
        res.headers["Cache-Control"] = "no-cache"
        res.headers["Connection"] = "keep-alive"
        id = OpenAIServer.completion_id
        created = Time.utc.to_unix

        send = ->(chunk : String) {
          res.print "data: #{chunk}\n\n"
          res.flush
        }
        send.call(OpenAIServer.chunk_json(id, model, created, delta_role: "assistant"))

        gen = engine.generate(request.messages, max_tokens, temperature, top_k, rep_pen, stops, request.seed, request.tool_defs, request.forced_tool) do |piece|
          send.call(OpenAIServer.chunk_json(id, model, created, delta_content: piece))
        end

        send.call(OpenAIServer.chunk_json(id, model, created, finish_reason: gen.finish_reason))
        res.print "data: [DONE]\n\n"
        res.flush
      else
        gen = engine.generate(request.messages, max_tokens, temperature, top_k, rep_pen, stops, request.seed, request.tool_defs, request.forced_tool) { }
        res.content_type = "application/json"
        res.print OpenAIServer.chat_completion_json(OpenAIServer.completion_id, model, gen)
      end
    else
      res.status = HTTP::Status::NOT_FOUND
      res.content_type = "application/json"
      res.print OpenAIServer.error_json("Unknown route: #{req.method} #{req.path}", "not_found")
    end
  ensure
    # One line per request, on EVERY exit path -- which is why it is an ensure rather than a call at the
    # end of each branch. A client that gets a 404 on a route this server does not implement, or a 400
    # on a body it rejected, previously left no trace at all: the failure was visible only to the
    # client, and from here the server looked idle. That is the first thing anyone needs when a client
    # will not talk to it.
    OpenAIServer::Log.info do
      "#{peer} #{req.method} #{req.path} -> #{res.status.code} " \
      "in #{(Time.instant - started).total_milliseconds.round(0).to_i}ms"
    end
  end
end

address = server.bind_tcp(host, port)
STDERR.puts "OpenAI-compatible API on http://#{address} (model: #{model_name})"
STDERR.puts "  POST /v1/chat/completions   GET /v1/models   GET /health"
STDERR.puts "  tools: OpenAI function calling (tools / tool_choice / role:\"tool\")"
STDERR.puts "  auth: #{api_key ? "Bearer token required (SHAINET_API_KEY)" : "none (localhost only)"}"
server.listen
