#!/usr/bin/env ruby
# frozen_string_literal: true

# Data root — where _sources/, _dataset/, etc. live. Override with
# LLM_TRAINER_ROOT (used by tests to run the pipeline in a sandbox).
ROOT = ENV["LLM_TRAINER_ROOT"] || File.expand_path("..", File.dirname(__FILE__)) unless defined?(ROOT)

# Build a plain-text corpus for continued pretraining (domain adaptation).
#
# Emits one {"text": "..."} JSON object per line — the format accepted by
# LLaMA-Factory's `pt` stage and by mlx-lm — containing:
#   - every .rb file under _sources/ (verbatim source, one repo at a time)
#   - every guide chapter from docs_context/ (front matter stripped and
#     split at chapter level, so documents fit model context windows)
#
# Documents that can't be processed (invalid encoding, unreadable files,
# malformed front matter) are skipped with a warning — the run never aborts
# on bad data. Duplicate documents (by content hash) are emitted only once.
#
# Usage:
#   ruby build/build_pretrain_corpus.rb [output.jsonl]
#
# Defaults to _pretrain/ruby_corpus.jsonl.

require "fileutils"
require "json"
require "digest"
require_relative "create_dataset"

SOURCES_ROOT = File.join(ROOT, "_sources")
DOCS_DIR     = File.join(ROOT, "docs_context")
OUTPUT       = File.expand_path(ARGV[0] || File.join(ROOT, "_pretrain", "ruby_corpus.jsonl"))

SKIP_DIRS = %w[.git node_modules vendor tmp log .bundle pkg].freeze

# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def rb_files_under(root)
  Dir.glob(File.join(root, "**", "*.rb"))
     .reject { |f| SKIP_DIRS.any? { |d| f.split(File::SEPARATOR).include?(d) } }
     .sort
end

# Every .rb file from every repository under _sources/, as [label, abs_path]
# pairs. The label is the repository name.
def collect_code_files
  files = []

  Dir.glob(File.join(SOURCES_ROOT, "*"))
     .select { |p| File.directory?(p) }
     .reject { |p| File.basename(p).start_with?(".") }
     .sort
     .each do |repo_root|
    repo = File.basename(repo_root)
    rb_files_under(repo_root).each { |f| files << [repo, f] }
  end

  files
end

# Guide chapters as [title, text] pairs (front matter stripped, one entry per
# chapter so each document fits a model context window). A guide that fails to
# parse (malformed YAML, unexpected front matter, ...) is skipped, not fatal.
def collect_guide_documents
  docs = []
  skipped = 0
  guides = Dir.glob(File.join(DOCS_DIR, "**", "*.md")).sort
              .reject { |f| File.basename(f) == "INDEX.md" }

  guides.each do |path|
    begin
      guide = parse_guide(path)
      unless guide[:title].is_a?(String) && guide[:body].is_a?(String)
        raise "unexpected guide structure"
      end
      sections = split_guide_sections(guide[:body])
    rescue StandardError => e
      skipped += 1
      puts "  [warn] skipping guide #{File.basename(path)}: #{e.class}: #{e.message}"
      next
    end

    sections.each do |section|
      heading = section[:heading].to_s.strip
      content = section[:content].strip
      next if heading.empty? || content.empty?
      docs << [guide[:title], "## #{heading}\n\n#{content}"]
    end
  end

  puts "  Guides skipped (unparseable): #{skipped}" if skipped.positive?
  docs
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Writes one document through the dedup + error-skip guards. Counters tracks
# :written, :dupes, :errors.
def emit_document(f, text, seen, counters, label)
  hash = Digest::SHA256.hexdigest(text)
  if seen[hash]
    counters[:dupes] += 1
    return
  end

  begin
    f.puts JSON.generate("text" => text)
  rescue StandardError => e
    counters[:errors] += 1
    puts "  [warn] skipping #{label}: #{e.class}: #{e.message}"
    return
  end

  seen[hash] = true
  counters[:written] += 1
  counters[:chars] += text.length
end

def main
  code_files = collect_code_files
  guides     = collect_guide_documents

  seen = {}
  counters = { written: 0, dupes: 0, errors: 0, chars: 0 }

  FileUtils.mkdir_p(File.dirname(OUTPUT))

  File.open(OUTPUT, "w:UTF-8") do |f|
    code_files.each do |label, abs|
      begin
        text = File.read(abs, encoding: "UTF-8")
      rescue StandardError => e
        counters[:errors] += 1
        puts "  [warn] skipping #{abs}: #{e.class}: #{e.message}"
        next
      end
      next if text.strip.empty?
      emit_document(f, text, seen, counters, abs)
    end

    guides.each do |title, text|
      emit_document(f, text, seen, counters, "guide chapter \"#{title}\"")
    end
  end

  puts "#{code_files.size} .rb files + #{guides.size} guide chapters found"
  puts "Duplicates skipped: #{counters[:dupes]}  |  unprocessable skipped: #{counters[:errors]}"
  puts "Wrote #{counters[:written]} documents (#{counters[:chars]} chars, ~#{counters[:chars] / 4} tokens) to #{OUTPUT}"
end

main if __FILE__ == $PROGRAM_NAME
