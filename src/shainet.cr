require "logger"
require "./shainet/**"
require "apatite"

module SHAInet
  include Apatite
  alias GenNum = Float64 | Int32 | Int64 | Float32
end
