# V4A context-diff patch format — parser + applier.
#
# Lifted from bar-bot (src/product/v4a.cr, MIT, same author), which ported it from
# bright-agent's TypeScript implementation. Patches locate edits by surrounding context (not
# line numbers), with progressively fuzzier matching.
#
# This is what makes multi-file edits one call instead of six edit_file round-trips, and
# context-located hunks survive the line drift that defeats line-numbered edits.
#
# Format:
#   *** Begin Patch
#   *** Update File: path/to/file.ts
#   @@ optional context anchor
#    unchanged context line
#   -removed line
#   +added line
#   *** Add File: path/to/new.ts
#   +content line
#   *** Delete File: path/to/old.ts
#   *** End Patch

require "./agent_workspace"

module AgentDemo
  module V4A
    record PatchChunk,
      change_contexts : Array(String),
      old_lines : Array(String),
      new_lines : Array(String),
      is_end_of_file : Bool

    record FileAction,
      action_type : String, # "Add", "Update", "Delete"
      path : String,
      chunks : Array(PatchChunk),
      new_content : String

    class PatchParseError < Exception; end

    class ApplyHunkError < Exception; end

    BEGIN_PATCH  = "*** Begin Patch"
    END_PATCH    = "*** End Patch"
    ADD_FILE     = "*** Add File: "
    DELETE_FILE  = "*** Delete File: "
    UPDATE_FILE  = "*** Update File: "
    END_OF_FILE  = "*** End of File"
    CONTEXT_MARK = "@@ "
    EMPTY_CTX    = "@@"

    # Parse a V4A patch string into file actions
    def self.parse_patch(content : String) : Array(FileAction)
      lines = content.strip.split("\n").map(&.rstrip('\r'))
      lines = ensure_markers(lines)
      validate_markers(lines)

      actions = [] of FileAction
      remaining = lines[1...-1] # between Begin and End
      line_num = 2

      while remaining.size > 0
        if remaining[0].strip.empty?
          remaining = remaining[1..]
          line_num += 1
          next
        end
        action, consumed = parse_one_action(remaining, line_num)
        actions << action
        line_num += consumed
        remaining = remaining[consumed..]
      end

      actions
    end

    # Apply a parsed patch to files in a working directory
    def self.apply(work_dir : String, patch_text : String) : {applied: Array(String), errors: Array(String)}
      actions = parse_patch(patch_text)
      applied = [] of String
      errors = [] of String

      actions.each do |action|
        path = WorkspacePath.resolve(work_dir, action.path)
        case action.action_type
        when "Add"
          Dir.mkdir_p(File.dirname(path))
          File.write(path, action.new_content)
          applied << "Added: #{action.path}"
        when "Delete"
          if File.exists?(path)
            File.delete(path)
            applied << "Deleted: #{action.path}"
          else
            errors << "Delete failed: #{action.path} not found"
          end
        when "Update"
          unless File.exists?(path)
            errors << "Update failed: #{action.path} not found"
            next
          end
          content = File.read(path)
          new_content = apply_update(content, action)
          File.write(path, new_content)
          applied << "Updated: #{action.path}"
        end
      rescue ex : ApplyHunkError
        errors << "Patch failed for #{action.path}: #{ex.message}"
      rescue ex
        errors << "Error on #{action.path}: #{ex.message}"
      end

      {applied: applied, errors: errors}
    end

    # Apply update hunks to file content
    def self.apply_update(content : String, action : FileAction) : String
      lines = content.split("\n")
      result = [] of String
      processed_up_to = 0

      action.chunks.each_with_index do |chunk, idx|
        search_start = advance_cursor_to_anchors(chunk.change_contexts, lines, processed_up_to)
        matches = find_section_positions(lines, chunk, search_start)

        if matches.empty?
          raise ApplyHunkError.new("Cannot locate hunk #{idx + 1}: context lines don't match file content")
        end
        if matches.size > 1
          raise ApplyHunkError.new("Hunk #{idx + 1} is ambiguous: matches #{matches.size} locations (lines #{matches.map { |m| m + 1 }.join(", ")}). Add more context.")
        end

        hunk_start = matches[0]
        result.concat(lines[processed_up_to...hunk_start])
        result.concat(chunk.new_lines)
        processed_up_to = hunk_start + chunk.old_lines.size
      end

      result.concat(lines[processed_up_to..]) if processed_up_to < lines.size
      result.join("\n")
    end

    # --- Parsing helpers ---

    private def self.ensure_markers(lines : Array(String)) : Array(String)
      result = [] of String
      result << BEGIN_PATCH unless lines.first?.try(&.strip) == BEGIN_PATCH
      result.concat(lines)
      result << END_PATCH unless lines.last?.try(&.strip) == END_PATCH
      result
    end

    private def self.validate_markers(lines : Array(String))
      raise PatchParseError.new("Patch too short") if lines.size < 2
      raise PatchParseError.new("Missing '*** Begin Patch'") unless lines.first.strip == BEGIN_PATCH
      raise PatchParseError.new("Missing '*** End Patch'") unless lines.last.strip == END_PATCH
    end

    private def self.parse_one_action(lines : Array(String), line_num : Int32) : {FileAction, Int32}
      first = lines[0].strip
      if first.starts_with?(ADD_FILE)
        parse_add_file(lines, first)
      elsif first.starts_with?(DELETE_FILE)
        parse_delete_file(first)
      elsif first.starts_with?(UPDATE_FILE)
        parse_update_file(lines, first)
      else
        raise PatchParseError.new("Invalid operation at line #{line_num}: '#{first}'")
      end
    end

    private def self.parse_add_file(lines : Array(String), first : String) : {FileAction, Int32}
      path = first[ADD_FILE.size..].strip
      raise PatchParseError.new("Missing path in Add File") if path.empty?

      contents = [] of String
      consumed = 1
      (1...lines.size).each do |i|
        break unless lines[i].starts_with?("+")
        contents << lines[i][1..]
        consumed += 1
      end

      action = FileAction.new("Add", path, [] of PatchChunk, contents.map { |c| "#{c}\n" }.join)
      {action, consumed}
    end

    private def self.parse_delete_file(first : String) : {FileAction, Int32}
      path = first[DELETE_FILE.size..].strip
      raise PatchParseError.new("Missing path in Delete File") if path.empty?
      {FileAction.new("Delete", path, [] of PatchChunk, ""), 1}
    end

    private def self.parse_update_file(lines : Array(String), first : String) : {FileAction, Int32}
      path = first[UPDATE_FILE.size..].strip
      raise PatchParseError.new("Missing path in Update File") if path.empty?

      remaining = lines[1..]
      consumed = 1
      chunks = [] of PatchChunk
      allow_missing_context = true

      while remaining.size > 0
        break if remaining[0].strip.starts_with?("***")
        if remaining[0].strip.empty? && !chunks.empty?
          remaining = remaining[1..]
          consumed += 1
          next
        end
        chunk, chunk_lines = parse_update_chunk(remaining, allow_missing_context)
        if chunk
          chunks << chunk
        end
        consumed += chunk_lines
        remaining = remaining[chunk_lines..]
        allow_missing_context = false
      end

      raise PatchParseError.new("Empty Update for '#{path}'") if chunks.empty?
      {FileAction.new("Update", path, chunks, ""), consumed}
    end

    private def self.parse_update_chunk(lines : Array(String), allow_missing_context : Bool) : {PatchChunk?, Int32}
      contexts, start_idx = parse_context_markers(lines)

      if contexts.empty? && !allow_missing_context
        # Not a valid chunk start
        return {nil, 1}
      end

      return {nil, start_idx} if start_idx >= lines.size

      old_lines = [] of String
      new_lines = [] of String
      is_eof = false
      consumed = 0

      (start_idx...lines.size).each do |i|
        line = lines[i]

        if line == END_OF_FILE
          is_eof = true
          consumed = i + 1
          break
        end

        # Stop at next context marker or file operation
        if consumed > 0
          break if line == EMPTY_CTX || line.starts_with?(CONTEXT_MARK)
          break if line.strip.starts_with?("***")
        end

        if line.empty?
          old_lines << ""
          new_lines << ""
          consumed = i + 1
        elsif line[0] == ' '
          old_lines << line[1..]
          new_lines << line[1..]
          consumed = i + 1
        elsif line[0] == '+'
          new_lines << (line.size > 1 ? line[1..] : "")
          consumed = i + 1
        elsif line[0] == '-'
          old_lines << (line.size > 1 ? line[1..] : "")
          consumed = i + 1
        else
          break if consumed > 0
          return {nil, 1}
        end
      end

      consumed = start_idx if consumed == 0
      chunk = PatchChunk.new(contexts, old_lines, new_lines, is_eof)
      {chunk, consumed}
    end

    private def self.parse_context_markers(lines : Array(String)) : {Array(String), Int32}
      contexts = [] of String
      i = 0
      while i < lines.size
        if lines[i] == EMPTY_CTX
          contexts << ""
        elsif lines[i].starts_with?(CONTEXT_MARK)
          contexts << lines[i][CONTEXT_MARK.size..].strip
        else
          break
        end
        i += 1
      end
      {contexts, i}
    end

    # --- Matching helpers ---

    private def self.advance_cursor_to_anchors(anchors : Array(String), lines : Array(String), cursor : Int32) : Int32
      c = cursor
      anchors.each { |a| c = advance_cursor_to_anchor(a, lines, c) }
      c
    end

    private def self.advance_cursor_to_anchor(anchor : String, lines : Array(String), cursor : Int32) : Int32
      return cursor if anchor.strip.empty?

      # Look for anchor after cursor (exact then trim match)
      (cursor...lines.size).each do |i|
        return i + 1 if lines[i] == anchor
      end
      (cursor...lines.size).each do |i|
        return i + 1 if lines[i].strip == anchor.strip
      end
      cursor
    end

    private def self.find_section_positions(lines : Array(String), chunk : PatchChunk, start_from : Int32) : Array(Int32)
      if chunk.old_lines.empty?
        return [chunk.is_end_of_file ? lines.size : Math.min(start_from, lines.size)]
      end

      if chunk.is_end_of_file
        end_start = Math.max(0, lines.size - chunk.old_lines.size)
        return [] of Int32 if end_start < start_from
        return find_context_matches_at(lines, chunk.old_lines, end_start)
      end

      find_context_matches(lines, chunk.old_lines, start_from)
    end

    private def self.find_context_matches_at(lines : Array(String), context : Array(String), start : Int32) : Array(Int32)
      return [start] if context.empty?
      return [] of Int32 if start < 0 || start + context.size > lines.size

      # Try exact, trim-end, trim match
      return [start] if equals_slice(lines, context, start) { |a, b| a == b }
      return [start] if equals_slice(lines, context, start) { |a, b| a.rstrip == b.rstrip }
      return [start] if equals_slice(lines, context, start) { |a, b| a.strip == b.strip }
      [] of Int32
    end

    private def self.find_context_matches(lines : Array(String), context : Array(String), start_from : Int32) : Array(Int32)
      return [Math.min(start_from, lines.size)] if context.empty?

      from = Math.max(0, start_from)
      max_start = lines.size - context.size
      return [] of Int32 if from > max_start

      # Try each matcher level (exact → trim-end → trim)
      matchers = [
        ->(a : String, b : String) { a == b },
        ->(a : String, b : String) { a.rstrip == b.rstrip },
        ->(a : String, b : String) { a.strip == b.strip },
      ]

      matchers.each do |matcher|
        matches = [] of Int32
        (from..max_start).each do |i|
          matches << i if equals_slice_proc(lines, context, i, matcher)
        end
        return matches unless matches.empty?
      end

      [] of Int32
    end

    private def self.equals_slice(lines : Array(String), context : Array(String), start : Int32, &) : Bool
      return false if start < 0 || start + context.size > lines.size
      context.each_with_index do |ctx_line, i|
        return false unless yield(lines[start + i], ctx_line)
      end
      true
    end

    private def self.equals_slice_proc(lines : Array(String), context : Array(String), start : Int32, matcher : Proc(String, String, Bool)) : Bool
      return false if start < 0 || start + context.size > lines.size
      context.each_with_index do |ctx_line, i|
        return false unless matcher.call(lines[start + i], ctx_line)
      end
      true
    end
  end
end
