require "./spec_helper"

EXAMPLES = {
  "examples/llm_sample.cr",
  "examples/transformer_lm.cr",
  "examples/transformer_pe.cr",
  "examples/hf_gpt_import.cr",
}

describe "Examples" do
  EXAMPLES.each do |ex|
    it "compiles #{ex}" do
      status = Process.run("crystal", ["build", "--no-codegen", ex], chdir: File.join(__DIR__, ".."), output: Process::Redirect::Pipe, error: Process::Redirect::Pipe)
      status.exit_code.should eq(0)
    end
  end
end
