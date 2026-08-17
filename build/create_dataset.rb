#!/usr/bin/env ruby
# frozen_string_literal: true

# Data root — where _sources/, _dataset/, etc. live. Override with
# LLM_TRAINER_ROOT (used by tests to run the pipeline in a sandbox).
ROOT = ENV["LLM_TRAINER_ROOT"] || File.expand_path("..", File.dirname(__FILE__)) unless defined?(ROOT)

# Build fine-tuning datasets from the docs and the per-repo code datasets.
#
# The code datasets (code_<repo>.jsonl, one per repository under _sources/)
# are produced by build_code_context.rb — this script treats them as input.
#
# For every guide in docs_context/**/*.md it emits
#   _dataset/docs_<guide>.jsonl
# with:
#   - one overview entry per guide (front-matter description + intro section)
#   - one entry per chapter (chapters are Setext-underlined headings or ATX
#     `##` headings; `###`/`####` subsections stay inside their chapter)
#
# A manifest _dataset/INDEX.md is written at the end, plus a single combined
# training file _dataset/_full_ruby_dataset.jsonl containing every dataset
# generated in the run (the exact same JSONL lines as the per-dataset files,
# concatenated), ready to feed to local training pipelines directly.
#
# Usage:
#   ruby create_dataset.rb [docs_context_dir] [output_dir]
#
# Defaults: docs_context/ _dataset/ relative to this script.

require "fileutils"
require "json"
require "yaml"
require_relative "build_dataset"

DOCS_CONTEXT_DIR = File.expand_path(ARGV[0] || File.join(ROOT, "docs_context"))
OUTPUT_DIR       = File.expand_path(ARGV[1] || File.join(ROOT, "_dataset"))

SKIP_FILES = %w[INDEX.md].freeze

# ---------------------------------------------------------------------------
# Code datasets: produced by build_code_context.rb, consumed here as rows
# ---------------------------------------------------------------------------

def code_dataset_rows(output_dir)
  rows = []
  Dir.glob(File.join(output_dir, "code_*.jsonl")).sort.each do |f|
    count = 0
    File.foreach(f) { count += 1 }
    rows << { file: File.basename(f), source: File.join("code_context", File.basename(f, ".jsonl") + ".md"), entries: count, bytes: File.size(f) }
  end
  if rows.empty?
    puts "  [warn] no code_*.jsonl datasets found — run build_code_context.rb first"
  end
  rows
end

# ---------------------------------------------------------------------------
# Docs context: one dataset per guide
# ---------------------------------------------------------------------------

HEADING_UNDERLINE = /^[=-]+\s*$/
ATX_H2            = /^## (.+)$/
LIST_ITEM_PREFIX  = /^\s*([-*+]|\d+\.)\s/

def parse_guide(path)
  content = File.read(path, encoding: "UTF-8")
  front = {}

  if content =~ /\A---\n(.*?)\n---\n/m
    front = YAML.safe_load(Regexp.last_match(1)) || {}
    content = content.sub(/\A---\n.*?\n---\n/m, "")
  end

  {
    title: front["title"] || File.basename(path, ".md"),
    category: front["category"],
    description: front["description"],
    work_in_progress: !!front["work_in_progress"],
    body: content,
  }
end

# Splits a guide body into sections at chapter headings. Chapters are either
# Setext headings (a text line underlined by --- or ===) or ATX `##` headings.
# Everything inside ``` fences is ignored. Returns [{ heading:, content: }];
# the first section is the doc title + intro.
def split_guide_sections(body)
  sections = []
  current_heading = nil
  current = []
  in_fence = false

  lines = body.lines
  lines.each_with_index do |line, i|
    # Fences may be indented (some converted guides close blocks with a
    # leading space), so match with optional leading whitespace.
    if line =~ /^\s*```/
      in_fence = !in_fence
      current << line
      next
    end

    if !in_fence && line =~ ATX_H2
      sections << { heading: current_heading, content: current.join } if current_heading
      current_heading = Regexp.last_match(1).strip
      current = []
      next
    end

    if !in_fence && line =~ HEADING_UNDERLINE && i > 0
      prev = lines[i - 1]
      # An underline is a heading only when the previous line is plain text
      # (not blank, not itself an underline, not a list item or fence).
      if prev.strip != "" && prev !~ HEADING_UNDERLINE && prev !~ LIST_ITEM_PREFIX && prev !~ /^\s*```/
        current.pop # drop the heading text line that was already appended
        sections << { heading: current_heading, content: current.join } if current_heading
        current_heading = prev.strip
        current = []
        next
      end
    end

    current << line
  end

  # An unterminated trailing fence (seen in two guides) simply ends at EOF.
  sections << { heading: current_heading, content: current.join } if current_heading
  sections
end

DOC_OVERVIEW_PROMPTS = [
  "What does the guide \"%s\" cover? Give an overview.",
  "Summarize the guide \"%s\".",
  "I want an overview of the guide \"%s\". What will I learn?",
].freeze

DOC_CHAPTER_PROMPTS = [
  "Explain \"%s\" from the guide \"%s\".",
  "Describe \"%s\" as covered in the guide \"%s\".",
  "Walk me through \"%s\" from the guide \"%s\".",
].freeze

def docs_entries(guide)
  sections = split_guide_sections(guide[:body])
  entries = []
  counter = 0

  # Overview: front-matter description + the intro (title section) content.
  intro = sections.empty? ? guide[:body] : sections.first[:content]
  answer = [guide[:description].to_s.strip, intro.strip].reject(&:empty?).join("\n\n")
  unless answer.empty?
    entries.concat(split_long_entries(format(DOC_OVERVIEW_PROMPTS[counter % DOC_OVERVIEW_PROMPTS.size], guide[:title]), answer))
    counter += 1
  end

  # One entry per chapter.
  sections.drop(1).each do |section|
    heading = section[:heading].to_s.strip
    content = section[:content].strip
    next if heading.empty? || content.empty?

    human = format(DOC_CHAPTER_PROMPTS[counter % DOC_CHAPTER_PROMPTS.size], heading, guide[:title])
    counter += 1
    entries.concat(split_long_entries(human, "## #{heading}\n\n#{content}"))
  end

  entries
end

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(path, entries)
  File.open(path, "w:UTF-8") do |f|
    entries.each do |human, gpt|
      f.puts JSON.generate(
        "conversations" => [
          { "from" => "human", "value" => human },
          { "from" => "gpt",   "value" => gpt },
        ],
      )
    end
  end
end

def human_size(bytes)
  format("%.1f KB", bytes / 1024.0)
end

# Formats an elapsed time (seconds) as e.g. "12.3s", "1m 05.2s", or "1h 02m 03.1s".
def format_duration(seconds)
  return format("%.1fs", seconds) if seconds < 60
  return format("%dm %04.1fs", seconds / 60, seconds % 60) if seconds < 3600

  format("%dh %02dm %04.1fs", seconds / 3600, (seconds % 3600) / 60, seconds % 60)
end

def build_manifest(rows, totals)
  lines = []
  lines << "# Dataset Index — code_context + docs_context"
  lines << ""
  lines << "ShareGPT-format JSONL datasets generated by `create_dataset.rb`."
  lines << ""
  lines << "| Dataset | Source | Entries | Size |"
  lines << "|---|---|---|---|"
  rows.each do |row|
    lines << "| `#{row[:file]}` | `#{row[:source]}` | #{row[:entries]} | #{human_size(row[:bytes])} |"
  end
  lines << "| **Total** | — | **#{totals[:entries]}** | **#{human_size(totals[:bytes])}** |"
  lines << ""
  lines << "## Entry types"
  lines << ""
  lines << "- `code_*.jsonl` — per source file: verbatim code reproduction, API"
  lines << "  explanation (from the real RDoc comments), test coverage listings"
  lines << "  (Rails `test`, minitest `def test_`, RSpec `it` — from test/ and spec/)."
  lines << "- `docs_*.jsonl` — per guide: one overview entry + one entry per chapter."
  lines << ""
  lines << "## Licensing"
  lines << ""
  lines << "Each dataset is derived from a repository under `_sources/` and carries"
  lines << "that repository's license — see the license file inside each repository."
  lines << "All licenses in use permit local training and local use of the derived"
  lines << "data; review them before publishing or serving a trained model."
  lines.join("\n")
end

# Concatenates every dataset written in this run into a single JSONL file.
# The file contains exactly the same lines as the per-dataset files, in the
# same order — one ShareGPT conversation object per line — so it can be fed
# directly to local training pipelines without any conversion.
#
# The write is atomic and self-validating: the content goes to a temp file,
# every line is parsed and checked against the per-dataset files, and only
# then does the temp file replace the previous one. The final file is always
# either the complete, valid output or the previous good file — never a
# truncated or malformed intermediate.
def write_full_dataset_jsonl(path, rows)
  tmp_path = "#{path}.tmp"

  File.open(tmp_path, "w:UTF-8") do |f|
    rows.each do |row|
      File.foreach(File.join(OUTPUT_DIR, row[:file])) { |line| f.write(line) }
    end
  end

  validate_full_dataset_jsonl(tmp_path, rows)
  File.rename(tmp_path, path)
end

def line_count(path)
  count = 0
  File.foreach(path) { count += 1 }
  count
end

# Parses every line of the combined JSONL file and asserts the ShareGPT
# structure. Aborts (keeping the temp file for inspection) on malformed lines
# or on a line-count mismatch against the per-dataset files.
def validate_full_dataset_jsonl(path, rows)
  expected = rows.sum { |row| line_count(File.join(OUTPUT_DIR, row[:file])) }

  count = 0
  problems = []
  File.foreach(path) do |line|
    count += 1
    begin
      convs = JSON.parse(line)["conversations"]
      raise "conversations must be a 2-turn array" unless convs.is_a?(Array) && convs.size == 2
      raise "turn roles must be human/gpt" unless convs[0]["from"] == "human" && convs[1]["from"] == "gpt"
      raise "empty turn value" if convs[0]["value"].to_s.strip.empty? || convs[1]["value"].to_s.strip.empty?
    rescue => e
      problems << "line #{count}: #{e.message}"
      break if problems.size >= 5
    end
  end

  unless problems.empty?
    abort "INVALID #{File.basename(path)}: #{problems.join("; ")} (temp file kept at #{path} for inspection)"
  end

  unless count == expected
    abort "INVALID #{File.basename(path)}: #{count} lines, expected #{expected} (temp file kept at #{path} for inspection)"
  end

  puts "  Validated #{count} lines in #{File.basename(path)}"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main
  abort "Docs context dir not found: #{DOCS_CONTEXT_DIR}" unless File.directory?(DOCS_CONTEXT_DIR)

  started_at = Time.now
  puts "Converting started at #{started_at.strftime("%Y-%m-%d %H:%M:%S")}"

  FileUtils.mkdir_p(OUTPUT_DIR)

  rows = code_dataset_rows(OUTPUT_DIR)
  total_entries = rows.sum { |r| r[:entries] }
  total_bytes = rows.sum { |r| r[:bytes] }
  rows.each { |r| puts "  #{r[:file]} — #{r[:entries]} entries (existing)" }

  doc_files = Dir.glob(File.join(DOCS_CONTEXT_DIR, "**", "*.md")).sort
                  .reject { |f| SKIP_FILES.include?(File.basename(f)) }

  wip_count = 0
  skipped = []
  written_docs = []

  doc_files.each do |f|
    rel = f.sub("#{DOCS_CONTEXT_DIR}/", "")
    parts = rel.split("/")
    repo = parts.length > 1 ? parts.first : "root"
    base = File.basename(f, ".md")

    guide = parse_guide(f)
    entries = docs_entries(guide)
    if entries.empty?
      skipped << f
      next
    end

    wip_count += 1 if guide[:work_in_progress]

    out = File.join(OUTPUT_DIR, "docs_#{repo}_#{base}.jsonl")
    write_jsonl(out, entries)
    written_docs << File.basename(out)
    rows << { file: File.basename(out), source: File.join("docs_context", rel), entries: entries.size, bytes: File.size(out) }
    total_entries += entries.size
    total_bytes += File.size(out)
    puts "  #{File.basename(out)} — #{entries.size} entries"
  end

  # Remove docs datasets whose source guides no longer exist.
  Dir.glob(File.join(OUTPUT_DIR, "docs_*.jsonl")).sort.each do |f|
    next if written_docs.include?(File.basename(f))
    puts "  [clean] removing stale #{File.basename(f)}"
    File.delete(f)
  end

  rows.sort_by! { |r| r[:file] }

  manifest = build_manifest(rows, entries: total_entries, bytes: total_bytes)
  File.write(File.join(OUTPUT_DIR, "INDEX.md"), manifest, encoding: "UTF-8")

  full_path = File.join(OUTPUT_DIR, "_full_ruby_dataset.jsonl")
  write_full_dataset_jsonl(full_path, rows)

  puts
  puts "Wrote #{rows.size} datasets (#{total_entries} entries, #{human_size(total_bytes)}, ~#{total_bytes / 4} tokens) to #{OUTPUT_DIR}"
  puts "Wrote #{File.basename(full_path)} (#{human_size(File.size(full_path))}) combining all datasets from this run"
  puts "WIP guides included: #{wip_count}"
  puts "Skipped: #{skipped.map { |f| File.basename(f) }.join(", ")}" unless skipped.empty?
  puts "Total time: #{format_duration(Time.now - started_at)}"
end

main if __FILE__ == $PROGRAM_NAME
