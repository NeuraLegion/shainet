require "log"
require "./shainet/**"
require "apatite"

module SHAInet
  Log = ::Log.for(self)
  include Apatite
  alias GenNum = Float64 | Int32 | Int64 | Float32

  ::Log.setup(:debug)
end
