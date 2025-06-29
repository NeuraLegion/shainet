require "log"
require "./shainet/**"

module SHAInet
  Log = ::Log.for(self)
  alias GenNum = Float64 | Int32 | Int64 | Float32

  ::Log.setup(:debug)
end
