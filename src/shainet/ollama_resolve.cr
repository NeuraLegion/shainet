require "json"

module SHAInet
  module OllamaResolve
    OLLAMA_MODELS = ::File.join(Path.home.to_s, ".ollama", "models")

    def self.resolve(name : String) : String?
      parts = name.split(":", 2)
      model = parts[0]
      tag = parts[1]? || "latest"

      manifest_path = ::File.join(OLLAMA_MODELS, "manifests", "registry.ollama.ai", "library", model, tag)
      return unless ::File.exists?(manifest_path)

      manifest = JSON.parse(::File.read(manifest_path))
      layers = manifest["layers"]?.try(&.as_a)
      return unless layers

      model_layer = layers.find { |l| l["mediaType"]?.try(&.as_s) == "application/vnd.ollama.image.model" }
      return unless model_layer

      digest = model_layer["digest"]?.try(&.as_s)
      return unless digest

      blob_path = ::File.join(OLLAMA_MODELS, "blobs", digest.gsub(":", "-"))
      ::File.exists?(blob_path) ? blob_path : nil
    end

    def self.ollama_name?(s : String) : Bool
      !s.includes?("/") && !s.includes?("\\") && !s.ends_with?(".gguf")
    end
  end
end
