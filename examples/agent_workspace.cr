# Workspace path confinement for the agent's file tools.
#
# Lifted from bar-bot (src/product/workspace_path.cr, MIT, same author) so the two agents
# share one guard rather than growing two subtly different ones.
#
# Every file tool resolves through this. Without it each tool took a raw path, and read_file
# has no confirmation gate, so a model could read ~/.aws/credentials or ~/.ssh/id_rsa
# unprompted and write anywhere in $HOME on a single "y".
#
# The rules that matter:
#   * absolute paths are refused outright, so the model must stay relative to the root
#   * realpath canonicalization, so a symlink cannot point out of the workspace
#   * for a path that does not exist yet, the nearest EXISTING ancestor is canonicalized and
#     re-checked -- otherwise a new file under a symlinked parent would escape
#   * .git is refused, since a tool rewriting git internals is never what was asked for
module AgentDemo
  class WorkspacePathError < Exception
  end

  module WorkspacePath
    extend self

    def resolve(root : String, relative : String) : String
      requested = relative.strip
      raise WorkspacePathError.new("path is empty") if requested.empty?
      raise WorkspacePathError.new("absolute paths are not allowed") if Path[requested].absolute?

      canonical_root = File.realpath(root)
      candidate = File.expand_path(requested, canonical_root)
      ensure_within_root!(canonical_root, candidate)
      reject_git_path!(canonical_root, candidate)

      existing = candidate
      until File.exists?(existing) || Dir.exists?(existing)
        parent = File.dirname(existing)
        raise WorkspacePathError.new("path escapes workspace") if parent == existing
        existing = parent
      end

      canonical_existing = File.realpath(existing)
      ensure_within_root!(canonical_root, canonical_existing)
      candidate
    rescue ex : WorkspacePathError
      raise ex
    rescue ex
      raise WorkspacePathError.new("invalid workspace path: #{ex.message}")
    end

    private def ensure_within_root!(root : String, candidate : String)
      return if candidate == root || candidate.starts_with?("#{root}/")
      raise WorkspacePathError.new("path escapes workspace")
    end

    private def reject_git_path!(root : String, candidate : String)
      relative = candidate == root ? "" : candidate[(root.size + 1)..]
      if relative.split('/').includes?(".git")
        raise WorkspacePathError.new(".git paths are managed by the worker")
      end
    end
  end
end
