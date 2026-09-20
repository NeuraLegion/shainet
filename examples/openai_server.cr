require "../src/shainet"
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
  extend self

  # ---- OpenAI request schema (the fields we honor; unknown fields ignored) ----

  struct ChatMessage
    include JSON::Serializable
    getter role : String
    getter content : String
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
  end

  # Result of one generation pass.
  record Generation, text : String, prompt_tokens : Int32,
    completion_tokens : Int32, finish_reason : String

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

    # Render an OpenAI message list into prompt token ids using whichever chat
    # template the tokenizer advertises (ChatML, then LLaMA 3, else a plain
    # concatenation), ending with the assistant generation prompt.
    def build_prompt(messages : Array(ChatMessage)) : Array(Int32)
      ids = [] of Int32
      im_start, im_end = @im_start, @im_end
      bos, start_hdr, end_hdr, eot = @bos, @start_hdr, @end_hdr, @eot

      if im_start && im_end
        messages.each do |m|
          ids << im_start
          ids.concat(@tok.encode(m.role))
          ids.concat(@nl)
          ids.concat(@tok.encode(m.content))
          ids << im_end
          ids.concat(@nl)
        end
        ids << im_start
        ids.concat(@tok.encode("assistant"))
        ids.concat(@nl)
      elsif bos && start_hdr && end_hdr && eot
        ids << bos
        messages.each do |m|
          ids << start_hdr
          ids.concat(@tok.encode(m.role))
          ids << end_hdr
          ids.concat(@nl2)
          ids.concat(@tok.encode(m.content))
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
                 stop_strings : Array(String), seed : Int64?, &block : String ->) : Generation
      @mutex.synchronize do
        @net.clear_cache!
        prompt_ids = build_prompt(messages)
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

        logits = @net.run(prompt_ids, stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
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

        Generation.new(text, prompt_ids.size, generated.size, finish)
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
                  j.field "content", gen.text
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

unless model_dir && Dir.exists?(model_dir)
  STDERR.puts "Usage: openai_server <model-dir> [port] [host]"
  STDERR.puts "  (point at an already-downloaded model, e.g. ~/models/Qwen3-0.6B)"
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
tokenizer = SHAInet::BPETokenizer.from_hf(File.join(model_dir, "tokenizer.json"))
STDERR.puts "Loaded in #{(Time.instant - t).total_seconds.round(1)}s (vocab #{tokenizer.vocab.size})"

model_name = File.basename(model_dir.rstrip("/"))
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

      gen = engine.generate(request.messages, max_tokens, temperature, top_k, rep_pen, stops, request.seed) do |piece|
        send.call(OpenAIServer.chunk_json(id, model, created, delta_content: piece))
      end

      send.call(OpenAIServer.chunk_json(id, model, created, finish_reason: gen.finish_reason))
      res.print "data: [DONE]\n\n"
      res.flush
    else
      gen = engine.generate(request.messages, max_tokens, temperature, top_k, rep_pen, stops, request.seed) { }
      res.content_type = "application/json"
      res.print OpenAIServer.chat_completion_json(OpenAIServer.completion_id, model, gen)
    end
  else
    res.status = HTTP::Status::NOT_FOUND
    res.content_type = "application/json"
    res.print OpenAIServer.error_json("Unknown route: #{req.method} #{req.path}", "not_found")
  end
end

address = server.bind_tcp(host, port)
STDERR.puts "OpenAI-compatible API on http://#{address} (model: #{model_name})"
STDERR.puts "  POST /v1/chat/completions   GET /v1/models   GET /health"
STDERR.puts "  auth: #{api_key ? "Bearer token required (SHAINET_API_KEY)" : "none (localhost only)"}"
server.listen
