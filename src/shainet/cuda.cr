module SHAInet
  module CUDA
    extend self

    # Check if CUDA runtime libraries can be opened.
    @@checked = false
    @@available = false

    def available?
      return @@available if @@checked
      @@checked = true
      handle = LibC.dlopen("libcudart.so", LibC::RTLD_LAZY)
      if handle.null?
        @@available = false
      else
        LibC.dlclose(handle)
        @@available = true
      end
      @@available
    rescue
      @@available = false
    end
  end
end
