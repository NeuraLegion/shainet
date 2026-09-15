require "./spec_helper"
require "file_utils"
require "../examples/agent_v4a"

# V4A is lifted code, so these examples are here to prove it behaves as the tool description
# promises IN THIS repo -- multi-file, multi-hunk, context-located, and honest about partial
# failure. A patch engine that silently half-applies is worse than one that refuses.
def with_workspace(&)
  dir = File.tempname("shainet_v4a")
  Dir.mkdir_p(dir)
  begin
    yield File.realpath(dir)
  ensure
    FileUtils.rm_rf(dir) if Dir.exists?(dir)
  end
end

describe AgentDemo::V4A do
  it "updates a file by surrounding context" do
    with_workspace do |ws|
      File.write(File.join(ws, "a.txt"), "one\ntwo\nthree\n")
      patch = <<-PATCH
        *** Begin Patch
        *** Update File: a.txt
        @@
         one
        -two
        +TWO
         three
        *** End Patch
        PATCH

      result = AgentDemo::V4A.apply(ws, patch)
      result[:errors].should be_empty
      # `applied` carries a human label per file, e.g. "Updated: a.txt", not the bare path.
      result[:applied].join(" ").should contain("a.txt")
      File.read(File.join(ws, "a.txt")).should eq("one\nTWO\nthree\n")
    end
  end

  it "adds a new file" do
    with_workspace do |ws|
      patch = <<-PATCH
        *** Begin Patch
        *** Add File: new/nested.txt
        +hello
        +world
        *** End Patch
        PATCH

      result = AgentDemo::V4A.apply(ws, patch)
      result[:errors].should be_empty
      File.read(File.join(ws, "new", "nested.txt")).should eq("hello\nworld\n")
    end
  end

  it "deletes a file" do
    with_workspace do |ws|
      File.write(File.join(ws, "gone.txt"), "bye\n")
      patch = <<-PATCH
        *** Begin Patch
        *** Delete File: gone.txt
        *** End Patch
        PATCH

      AgentDemo::V4A.apply(ws, patch)[:errors].should be_empty
      File.exists?(File.join(ws, "gone.txt")).should be_false
    end
  end

  it "changes several files in ONE call, which is the whole point over edit_file" do
    with_workspace do |ws|
      File.write(File.join(ws, "x.txt"), "alpha\nbeta\n")
      File.write(File.join(ws, "y.txt"), "gamma\ndelta\n")
      patch = <<-PATCH
        *** Begin Patch
        *** Update File: x.txt
        @@
         alpha
        -beta
        +BETA
        *** Update File: y.txt
        @@
         gamma
        -delta
        +DELTA
        *** End Patch
        PATCH

      result = AgentDemo::V4A.apply(ws, patch)
      result[:errors].should be_empty
      result[:applied].size.should eq(2)
      File.read(File.join(ws, "x.txt")).should eq("alpha\nBETA\n")
      File.read(File.join(ws, "y.txt")).should eq("gamma\nDELTA\n")
    end
  end

  it "reports a hunk whose context does not match instead of writing something wrong" do
    with_workspace do |ws|
      File.write(File.join(ws, "a.txt"), "one\ntwo\nthree\n")
      patch = <<-PATCH
        *** Begin Patch
        *** Update File: a.txt
        @@
         nonexistent context line
        -not here either
        +replacement
        *** End Patch
        PATCH

      result = AgentDemo::V4A.apply(ws, patch)
      result[:errors].should_not be_empty
      # Unchanged: a failed match must leave the file exactly as it was.
      File.read(File.join(ws, "a.txt")).should eq("one\ntwo\nthree\n")
    end
  end

  it "refuses a path escaping the workspace" do
    with_workspace do |ws|
      patch = <<-PATCH
        *** Begin Patch
        *** Add File: ../escaped.txt
        +planted
        *** End Patch
        PATCH

      result = AgentDemo::V4A.apply(ws, patch)
      result[:applied].should be_empty
      result[:errors].should_not be_empty
      File.exists?(File.join(File.dirname(ws), "escaped.txt")).should be_false
    end
  end

  it "raises on a patch with no markers rather than guessing" do
    with_workspace do |ws|
      expect_raises(AgentDemo::V4A::PatchParseError) do
        AgentDemo::V4A.apply(ws, "just some text\nwith no markers\n")
      end
    end
  end

  it "applies two hunks to the same file" do
    with_workspace do |ws|
      File.write(File.join(ws, "m.txt"), "a\nb\nc\nd\ne\nf\ng\nh\n")
      patch = <<-PATCH
        *** Begin Patch
        *** Update File: m.txt
        @@
         a
        -b
        +B
         c
        @@
         f
        -g
        +G
         h
        *** End Patch
        PATCH

      result = AgentDemo::V4A.apply(ws, patch)
      result[:errors].should be_empty
      File.read(File.join(ws, "m.txt")).should eq("a\nB\nc\nd\ne\nf\nG\nh\n")
    end
  end
end
