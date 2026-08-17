#!/usr/bin/env ruby
# frozen_string_literal: true

# Data root — where _sources/, _dataset/, etc. live. Override with
# LLM_TRAINER_ROOT (used by tests to run the pipeline in a sandbox).
ROOT = ENV["LLM_TRAINER_ROOT"] || File.expand_path("..", File.dirname(__FILE__)) unless defined?(ROOT)

# Build code-context files from every repository under _sources/.
#
# Fully source-agnostic: it scans _sources/*, runs `git pull` inside each repo
# (a failed pull is only a warning — the build continues with what's on
# disk), walks all .rb files in the repo, and produces:
#
#   code_context/<repo>.md      — agent-friendly source dump (one per repo)
#   _dataset/code_<repo>.jsonl  — ShareGPT dataset (one per repo)
#
# No folder structure inside a repo is assumed — any repo, however organized,
# is parsed the same way. Stale outputs (for repos that no longer exist under
# _sources/) are removed automatically.
#
# Usage:
#   ruby build/build_code_context.rb [sources_dir]
#
# Defaults to _sources/ relative to this script.

require "fileutils"
require "json"
require "set"
require_relative "build_dataset"

SOURCES_ROOT     = File.expand_path(ARGV[0] || File.join(ROOT, "_sources"))
OUTPUT_MD_DIR    = File.join(ROOT, "code_context")
OUTPUT_JSONL_DIR = File.join(ROOT, "_dataset")

SKIP_DIRS = %w[.git node_modules vendor tmp log .bundle pkg].to_set.freeze

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def human_size(bytes)
  if bytes < 1024
    "#{bytes} B"
  elsif bytes < 1024 * 1024
    format("%.1f KB", bytes / 1024.0)
  else
    format("%.1f MB", bytes / (1024.0 * 1024))
  end
end

# NOTE: do not use `tr("/._", "---")` here — Ruby's tr parses the replacement
# string as a range, and the `_` mapping differs between Ruby versions
# (≤3.1 maps `_` to `.`), so anchors flip between runs. gsub is explicit and
# version-independent, always producing all-dash anchors.
def anchor_for(path)
  path.gsub(/[\/._]/, "-")
end

# Runs `git pull` inside a source repo. A failed pull is only a warning — the
# build continues with whatever is already on disk.
def git_pull(repo_root)
  name = File.basename(repo_root)
  puts "  Updating #{name} (git pull)..."
  Dir.chdir(repo_root) do
    ok = system({ "GIT_EDITOR" => "true" }, "git", "pull")
    warn "  [warn] `git pull` failed for #{name} — continuing with the existing sources (they may be stale)." unless ok
  end
end

# ---------------------------------------------------------------------------
# Discovery & file walking (source-agnostic)
# ---------------------------------------------------------------------------

def discover_repos(sources_root)
  Dir.glob(File.join(sources_root, "*"))
     .select { |p| File.directory?(p) }
     .reject { |p| File.basename(p).start_with?(".") }
     .sort
     .map { |p| [File.basename(p), p] }
end

def rb_files(repo_root)
  Dir.glob(File.join(repo_root, "**", "*.rb"))
     .reject { |f| SKIP_DIRS.intersect?(f.split(File::SEPARATOR).to_set) }
     .sort
     .map { |abs| [abs.sub("#{repo_root}/", ""), abs] }
end

# ---------------------------------------------------------------------------
# Markdown generation (per repo)
# ---------------------------------------------------------------------------

def build_repo_md(repo, files)
  total_bytes = files.sum { |_, abs| File.size(abs) }

  lines = []
  lines << "# #{repo} — Ruby Source Code"
  lines << ""
  lines << "**#{files.size} files** — **#{human_size(total_bytes)}** total"
  lines << ""
  lines << "---"
  lines << ""
  lines << "## File Index"
  lines << ""

  files.each do |rel, abs|
    size = human_size(File.size(abs))
    lines << "- [`#{rel}`](##{anchor_for(rel)}) (#{size})"
  end

  lines << ""
  lines << "---"
  lines << ""

  files.each do |rel, abs|
    anchor = anchor_for(rel)
    content = File.read(abs, encoding: "UTF-8") rescue "# [Error reading file]"

    lines << %(<a id="#{anchor}"></a>)
    lines << ""
    lines << "## `#{rel}`"
    lines << ""
    lines << "```ruby"
    lines << content.rstrip
    lines << "```"
    lines << ""
  end

  lines.join("\n")
end

def build_index(repo_data)
  lines = []
  lines << "# Code Context — Index"
  lines << ""
  lines << "This directory contains the full Ruby source code of every project"
  lines << "under _sources/, one file per project, for use as context with AI"
  lines << "coding agents. Adding a repository to _sources/ and re-running"
  lines << "build_code_context.rb picks it up automatically."
  lines << ""
  lines << "| Project | Files | Size |"
  lines << "|---|---|---|"

  grand_files = 0
  grand_bytes = 0

  repo_data.each do |name, count, bytes|
    grand_files += count
    grand_bytes += bytes
    lines << "| [`#{name}.md`](#{name}.md) | #{count} | #{human_size(bytes)} |"
  end

  lines << "| **Total** | **#{grand_files}** | **#{human_size(grand_bytes)}** |"
  lines << ""
  lines << "---"
  lines << ""
  lines << "## Usage"
  lines << ""
  lines << "To set up an AI agent with code context:"
  lines << ""
  lines << "1. Point the agent at `code_context/INDEX.md` so it knows what's available."
  lines << "2. Have it read the `.md` file(s) relevant to the task."
  lines << "3. Also share `AGENTS.md` for coding conventions and architectural guidance."
  lines.join("\n")
end

# ---------------------------------------------------------------------------
# Dataset generation (per repo, ShareGPT format)
# ---------------------------------------------------------------------------

def build_repo_jsonl(repo, files)
  entries = []
  counter = 0
  skipped = 0

  files.each do |rel, abs|
    begin
      code = File.read(abs, encoding: "UTF-8")
    rescue StandardError
      skipped += 1
      next
    end
    next if code.strip.empty?

    is_test = test_file?(rel)

    # 1. Code reproduction entry
    pool = is_test ? TEST_PROMPTS : CODE_PROMPTS
    entries.concat(split_long_entries(format(pool[counter % pool.size], rel), code))
    counter += 1

    if is_test
      # 2. Test coverage entry
      coverage = build_coverage(rel, code)
      if coverage
        entries.concat(split_long_entries(format(COVERAGE_PROMPTS[counter % COVERAGE_PROMPTS.size], rel), coverage))
        counter += 1
      end
    else
      # 2. API explanation entry
      explanation = build_explanation(rel, code)
      if explanation
        entries.concat(split_long_entries(format(EXPLAIN_PROMPTS[counter % EXPLAIN_PROMPTS.size], rel), explanation))
        counter += 1
      end
    end
  end

  [entries, skipped]
end

def write_jsonl(path, entries)
  skipped = 0
  File.open(path, "w:UTF-8") do |f|
    entries.each do |human, gpt|
      begin
        f.puts JSON.generate(
          "conversations" => [
            { "from" => "human", "value" => human },
            { "from" => "gpt",   "value" => gpt },
          ],
        )
      rescue StandardError => e
        skipped += 1
        puts "  [warn] skipping unencodable entry: #{e.class}: #{e.message}"
      end
    end
  end
  skipped
end

def validate_jsonl(path)
  bad = 0
  count = 0
  File.foreach(path) do |line|
    count += 1
    begin
      convs = JSON.parse(line)["conversations"]
      raise "structure" unless convs.is_a?(Array) && convs.size == 2
      raise "roles" unless convs[0]["from"] == "human" && convs[1]["from"] == "gpt"
      raise "empty" if convs[0]["value"].to_s.strip.empty? || convs[1]["value"].to_s.strip.empty?
    rescue => e
      bad += 1
      puts "  [invalid] line #{count}: #{e.message}"
      break if bad >= 5
    end
  end
  abort "INVALID output: #{bad} bad lines in #{path}" unless bad.zero?
  count
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main
  abort "Sources dir not found: #{SOURCES_ROOT}" unless File.directory?(SOURCES_ROOT)

  repos = discover_repos(SOURCES_ROOT)
  abort "No repositories found under #{SOURCES_ROOT}" if repos.empty?

  FileUtils.mkdir_p(OUTPUT_MD_DIR)
  FileUtils.mkdir_p(OUTPUT_JSONL_DIR)

  repo_names = repos.map(&:first)

  # Remove outputs for repos that no longer exist under _sources/.
  stale_md = Dir.glob(File.join(OUTPUT_MD_DIR, "*.md"))
                 .reject { |f| File.basename(f) == "INDEX.md" }
                 .reject { |f| repo_names.include?(File.basename(f, ".md")) }
  stale_jsonl = Dir.glob(File.join(OUTPUT_JSONL_DIR, "code_*.jsonl"))
                   .reject { |f| repo_names.include?(File.basename(f, ".jsonl").sub(/\Acode_/, "")) }

  stale_md.each do |f|
    puts "  [clean] removing stale #{File.basename(f)}"
    File.delete(f)
  end
  stale_jsonl.each do |f|
    puts "  [clean] removing stale #{File.basename(f)}"
    File.delete(f)
  end

  repo_data = []
  total_entries = 0

  repos.each do |repo, repo_root|
    puts "Processing #{repo}..."
    git_pull(repo_root)

    files = rb_files(repo_root)
    if files.empty?
      puts "  [skip] #{repo} — no .rb files"
      next
    end

    total_bytes = files.sum { |_, abs| File.size(abs) }
    repo_data << [repo, files.size, total_bytes]

    puts "  Building #{repo}.md (#{files.size} files, #{human_size(total_bytes)})..."
    content = build_repo_md(repo, files)
    File.write(File.join(OUTPUT_MD_DIR, "#{repo}.md"), content, encoding: "UTF-8")

    entries, unreadable = build_repo_jsonl(repo, files)
    out = File.join(OUTPUT_JSONL_DIR, "code_#{repo}.jsonl")
    write_skipped = write_jsonl(out, entries)
    validated = validate_jsonl(out)
    total_entries += validated
    puts "  Wrote code_#{repo}.jsonl (#{validated} entries, unreadable skipped: #{unreadable}, unencodable skipped: #{write_skipped})"
  end

  repo_data.sort_by! { |_, _, bytes| -bytes }

  puts
  puts "  Building INDEX.md (#{repo_data.size} repos)..."
  index_content = build_index(repo_data)
  File.write(File.join(OUTPUT_MD_DIR, "INDEX.md"), index_content, encoding: "UTF-8")

  grand_files = repo_data.sum { |_, count, _| count }
  grand_bytes = repo_data.sum { |_, _, bytes| bytes }
  puts
  puts "Done! #{repo_data.size} repos, #{grand_files} files (#{human_size(grand_bytes)}) in code_context/, #{total_entries} dataset entries in _dataset/"
end

main if __FILE__ == $PROGRAM_NAME
