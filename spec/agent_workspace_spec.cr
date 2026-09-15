require "./spec_helper"
require "file_utils"
require "../examples/agent_workspace"

# The path guard is the only thing standing between a model-supplied string and the whole
# filesystem, and read_file has no confirmation prompt, so a hole here is silent. Every
# example below is an escape that a plausible model output would attempt.
#
# agent_workspace.cr is required directly rather than through agent.cr: that file has
# top-level code that starts a REPL, so requiring it from a spec would hang the suite. Keeping
# the guard in its own file is what makes it testable at all.
# A fresh root per example, since realpath is resolved against it.
def with_root(&)
  dir = File.tempname("shainet_root")
  Dir.mkdir_p(File.join(dir, "sub"))
  File.write(File.join(dir, "keep.txt"), "hello\n")
  begin
    yield File.realpath(dir)
  ensure
    FileUtils.rm_rf(dir) if Dir.exists?(dir)
  end
end

describe AgentDemo::WorkspacePath do
  it "resolves a plain relative path inside the root" do
    with_root do |root|
      AgentDemo::WorkspacePath.resolve(root, "keep.txt").should eq(File.join(root, "keep.txt"))
    end
  end

  it "resolves a path that does not exist yet, so new files can be created" do
    with_root do |root|
      AgentDemo::WorkspacePath.resolve(root, "sub/new.txt").should eq(File.join(root, "sub", "new.txt"))
    end
  end

  it "refuses an absolute path" do
    with_root do |root|
      expect_raises(AgentDemo::WorkspacePathError, /absolute/) do
        AgentDemo::WorkspacePath.resolve(root, "/etc/passwd")
      end
    end
  end

  it "refuses a traversal out of the root" do
    with_root do |root|
      expect_raises(AgentDemo::WorkspacePathError, /escapes workspace/) do
        AgentDemo::WorkspacePath.resolve(root, "../outside.txt")
      end
    end
  end

  it "refuses a traversal that only escapes after descending" do
    with_root do |root|
      expect_raises(AgentDemo::WorkspacePathError, /escapes workspace/) do
        AgentDemo::WorkspacePath.resolve(root, "sub/../../outside.txt")
      end
    end
  end

  it "refuses an empty path" do
    with_root do |root|
      expect_raises(AgentDemo::WorkspacePathError, /empty/) do
        AgentDemo::WorkspacePath.resolve(root, "   ")
      end
    end
  end

  it "refuses .git at any depth" do
    with_root do |root|
      expect_raises(AgentDemo::WorkspacePathError, /\.git/) do
        AgentDemo::WorkspacePath.resolve(root, ".git/config")
      end
      expect_raises(AgentDemo::WorkspacePathError, /\.git/) do
        AgentDemo::WorkspacePath.resolve(root, "sub/.git/hooks/pre-commit")
      end
    end
  end

  it "refuses a symlink pointing out of the root" do
    with_root do |root|
      outside = File.tempname("shainet_outside")
      Dir.mkdir_p(outside)
      begin
        File.write(File.join(outside, "secret.txt"), "token\n")
        # The classic escape: a link inside the workspace whose target is not.
        File.symlink(outside, File.join(root, "link"))
        expect_raises(AgentDemo::WorkspacePathError, /escapes workspace/) do
          AgentDemo::WorkspacePath.resolve(root, "link/secret.txt")
        end
      ensure
        FileUtils.rm_rf(outside)
      end
    end
  end

  it "refuses a NEW path under a symlinked parent" do
    with_root do |root|
      outside = File.tempname("shainet_outside2")
      Dir.mkdir_p(outside)
      begin
        File.symlink(outside, File.join(root, "link"))
        # The file does not exist, so the check must walk up to the nearest EXISTING ancestor
        # and canonicalize THAT. Checking only the literal path would let this through and a
        # write would land outside the workspace.
        expect_raises(AgentDemo::WorkspacePathError, /escapes workspace/) do
          AgentDemo::WorkspacePath.resolve(root, "link/planted.txt")
        end
      ensure
        FileUtils.rm_rf(outside)
      end
    end
  end

  it "allows the root itself" do
    with_root do |root|
      AgentDemo::WorkspacePath.resolve(root, ".").should eq(root)
    end
  end
end
