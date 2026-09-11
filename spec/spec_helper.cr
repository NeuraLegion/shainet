require "spec"
require "../src/shainet.cr"

# Several specs disable CUDA to build a CPU reference by setting the global
# SHAINET_DISABLE_CUDA flag. Some never restore it, and others restore it on a
# line that sits *after* their own `pending!` guard, so once the flag is set that
# guard aborts the example and the cleanup never runs. The flag is consulted on
# every availability query, so a single leak silently pends every GPU spec that
# happens to run later: the suite still reports success while quietly skipping
# GPU coverage, and which specs get skipped depends on file load order.
#
# Clearing it before each example makes that whole class of leak impossible.
# Specs that set it mid-example to compare against the CPU path are unaffected.
Spec.before_each { ENV.delete("SHAINET_DISABLE_CUDA") }

# Extract train data
