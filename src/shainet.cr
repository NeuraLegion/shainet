require "log"
require "./shainet/**"

module SHAInet
  Log = ::Log.for(self)
  alias GenNum = Float64 | Int32 | Int64 | Float32

  lvl = {
    "info"  => ::Log::Severity::Info,
    "debug" => ::Log::Severity::Debug,
    "warn"  => ::Log::Severity::Warn,
    "error" => ::Log::Severity::Error,
    "fatal" => ::Log::Severity::Fatal,
    "trace" => ::Log::Severity::Trace,
  }

  log_level = (ENV["LOG_LEVEL"]? || "info") # Default to info level if not set

  ::Log.setup(lvl[log_level.downcase])
end
