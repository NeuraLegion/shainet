require "json"

module SHAInet
  module PyTorchImport
    # Returns JSON data describing sequential linear layers.
    def self.load(file_path : String) : JSON::Any
      script = File.join(__DIR__, "../../scripts/pt_to_json.py")
      output = IO::Memory.new
      status = Process.run(
        "python3",
        [script, file_path],
        output: output,
        error: Process::Redirect::Close
      )
      raise "Failed to convert TorchScript" unless status.success?
      JSON.parse(output.to_s)
    end
  end
end
