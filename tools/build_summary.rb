#!/usr/bin/env ruby
# frozen_string_literal: true

# Distills a source repository into its most important knowledge as a compact
# ShareGPT JSONL file — a "memory capsule" for later refresh stages.
#
# When sources are replaced between training stages, the model's knowledge of
# the old source slowly decays. This tool captures the distilled essence of a
# repository (purpose + public API of every lib file, from the real RDoc
# comments) so it can be re-trained later to refresh that knowledge — without
# keeping the full source around.
#
# Output: one ShareGPT conversation per line:
#   {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
#
# Usage:
#   ruby tools/build_summary.rb <repo_dir> [output_dir] [--max-entries N]
#
# Defaults: output_dir = _tmp_sources/summary, max entries = 80.
# The output file is named after the repo directory (<repo>.jsonl).

require "fileutils"
require "json"
require_relative "../build/build_dataset"

POS_ARGS = ARGV.reject { |a| a.start_with?("--") }
FLAGS    = ARGV.select { |a| a.start_with?("--") }

REPO_DIR = File.expand_path(POS_ARGS[0] || ".")
OUT_DIR  = File.expand_path(POS_ARGS[1] || File.join(__dir__, "..", "_tmp_sources", "summary"))
MAX_ENTRIES = begin
  flag = FLAGS.find { |a| a.start_with?("--max-entries=") }
  flag ? Integer(flag.split("=", 2)[1]) : 80
end

DISTILL_PROMPTS = [
  "Quick reference: what does the file `%s` provide? Give its distilled API.",
  "Summarize the purpose and public API of `%s`.",
  "What are the most important classes and methods in `%s`?",
].freeze

# Returns [[human, answer], ...] — one distilled Q&A per lib file with an
# extractable API. Test files are skipped: the capsule keeps the knowledge,
# not the test suite.
def build_entries(repo_dir)
  entries = []
  files = Dir.glob(File.join(repo_dir, "**", "*.rb"))
             .reject { |f| f.include?("/test/") || f.include?("/spec/") }
             .sort

  files.each_with_index do |abs, i|
    rel = abs.sub("#{repo_dir}/", "")
    code = File.read(abs, encoding: "UTF-8")
    next if code.strip.empty?

    answer = build_explanation(rel, code)
    next if answer.nil?

    human = format(DISTILL_PROMPTS[i % DISTILL_PROMPTS.size], rel)
    entries << [human, answer]
  end
  entries
end

abort "Repo dir not found: #{REPO_DIR}" unless File.directory?(REPO_DIR)

entries = build_entries(REPO_DIR)
entries = entries.uniq { |_, answer| answer }[0, MAX_ENTRIES]

abort "No distillable API found in #{REPO_DIR}" if entries.empty?

FileUtils.mkdir_p(OUT_DIR)
out = File.join(OUT_DIR, "#{File.basename(REPO_DIR)}.jsonl")
File.open(out, "w:UTF-8") do |f|
  entries.each do |human, gpt|
    f.puts JSON.generate(
      "conversations" => [
        { "from" => "human", "value" => human },
        { "from" => "gpt",   "value" => gpt },
      ],
    )
  end
end

puts "Distilled #{entries.size} entries from #{File.basename(REPO_DIR)} (#{File.basename(out)}, #{format('%.1f', File.size(out) / 1024.0)} KB)"
