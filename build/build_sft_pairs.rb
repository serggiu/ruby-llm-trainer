#!/usr/bin/env ruby
# frozen_string_literal: true

# Data root — where _sources/, _dataset/, etc. live. Override with
# LLM_TRAINER_ROOT (used by tests to run the pipeline in a sandbox).
ROOT = ENV["LLM_TRAINER_ROOT"] || File.expand_path("..", File.dirname(__FILE__)) unless defined?(ROOT)

# Build the task-oriented SFT (supervised fine-tuning) training set for fine-tuning a local LLM on
# Ruby.
#
# The output (ShareGPT JSONL, same format as the other datasets) combines:
#   1. base entries   — every entry from _dataset/*.jsonl, deduplicated
#   2. method pairs   — "implement this method from its API documentation":
#                       prompt = RDoc comment + signature, answer = the real
#                       method body (only methods with doc comments)
#   3. test pairs     — "implement the class under test": prompt = test file,
#                       answer = the lib/app file it tests (matched by path)
#   4. bug-fix pairs  — "fix this bug": real fixes mined from the git history
#                       of every repository under _sources/ (single-file,
#                       small, fix-keyword commits); answer = the fixed file
#   5. guide pairs    — how-to Q&A from the guides with diversified
#                       natural-language prompts
#
# Robustness: any item that can't be processed (unreadable file, malformed
# JSON/YAML, truncated method capture, git hiccup, ...) is skipped with a
# warning and counted — the run never aborts on bad data, and generated
# pairs are deduplicated by answer content.
#
# Usage:
#   ruby build/build_sft_pairs.rb [dataset_dir] [output.jsonl] [--skip-bugs]
#
# Defaults: _dataset → _dataset/sft_train_set.jsonl

require "fileutils"
require "json"
require "digest"
require "open3"
require_relative "create_dataset"

POS_ARGS   = ARGV.reject { |a| a.start_with?("--") }
FLAGS      = ARGV.select { |a| a.start_with?("--") }

DATASET_DIR = File.expand_path(POS_ARGS[0] || File.join(ROOT, "_dataset"))
OUTPUT      = File.expand_path(POS_ARGS[1] || File.join(DATASET_DIR, "sft_train_set.jsonl"))
SKIP_BUGS   = FLAGS.include?("--skip-bugs")

CODE_DIR = File.join(ROOT, "code_context")
DOCS_DIR = File.join(ROOT, "docs_context")
SOURCES_ROOT = File.join(ROOT, "_sources")

# ---------------------------------------------------------------------------
# 1. Base entries (deduplicated)
# ---------------------------------------------------------------------------

GUIDE_PROMPT_RE = /^(Explain|Describe|Walk me through) "|^What does the guide|^Summarize the guide|^I want an overview/

def load_base_entries(dataset_dir)
  entries = []
  skipped = 0
  Dir.glob(File.join(dataset_dir, "*.jsonl")).sort.each do |f|
    base = File.basename(f)
    next if base == "_full_ruby_dataset.jsonl" || base == "sft_train_set.jsonl"

    File.foreach(f) do |line|
      begin
        convs = JSON.parse(line)["conversations"]
        unless convs.is_a?(Array) && convs.size == 2 &&
               convs[0].is_a?(Hash) && convs[1].is_a?(Hash) &&
               convs[0]["value"].is_a?(String) && convs[1]["value"].is_a?(String)
          raise "unexpected entry shape"
        end
        entries << [convs[0]["value"], convs[1]["value"]]
      rescue StandardError => e
        skipped += 1
        puts "  [warn] skipping bad entry in #{base}: #{e.message}"
      end
    end
  end
  puts "  Base entries skipped (unparseable): #{skipped}" if skipped.positive?

  # 1a. Exact duplicates (same human + same answer).
  exact = {}
  entries = entries.reject do |h, g|
    key = "#{h}\x00#{g}"
    exact[key] ? true : (exact[key] = true; false)
  end

  # 1b. Duplicate verbatim-code answers (e.g. files copied between lib/ and
  # test/). Guide Q&A entries are exempt — their prompts are intentionally
  # diversified later while the answers repeat chapter text.
  answers = {}
  entries = entries.reject do |h, g|
    next false if h =~ GUIDE_PROMPT_RE
    key = Digest::SHA256.hexdigest(g)
    answers[key] ? true : (answers[key] = true; false)
  end

  entries
end

# ---------------------------------------------------------------------------
# 2. Method-implementation pairs
# ---------------------------------------------------------------------------

DEF_RE       = /^\s*def\s+(?:self\.)?([a-zA-Z_]\w*[!?=]?)(\s*\([^;]*\))?/
DEF_SELF_RE  = /^\s*def\s+self\./
CLASS_RE     = /^\s*(class|module)\s+([A-Z]\w*)\b/
SELF_CLASS_RE = /^\s*class\s*<<\s*self\b/

# Block opener at the start of a line (block form, not a trailing modifier).
BLOCK_OPENER = /^\s*(if|unless|while|until|case|begin|class|module)\b/
DEF_OPENER   = /^\s*def\b/

# Naive, string-blind keyword balance. Good enough to reject truncated method
# captures (e.g. stopping at a nested `end` at the same indentation); a false
# rejection just skips the method, which is the safe choice.
def balanced_span?(span_lines)
  balance = 0
  span_lines.each do |l|
    balance += 1 if l =~ BLOCK_OPENER || l =~ DEF_OPENER
    balance += l.scan(/\bdo\b/).size
    balance += l.count("{")
    balance -= l.scan(/\bend\b/).size
    balance -= l.count("}")
  end
  balance.zero?
end

# Captures a method body: the def line through the first `end` at the def's
# own indentation whose keyword balance returns to zero. Handles one-liners
# (`def x; end`) and endless methods (`def x = expr`). Returns nil when no
# trustworthy end is found — the caller skips the method.
def capture_def_body(lines, def_index, def_indent, def_line)
  stripped = def_line.sub(/\([^)]*\)/, "")
  return def_line if def_line.include?(";") || stripped.include?("=")

  last = [def_index + 400, lines.size - 1].min
  (def_index + 1..last).each do |j|
    l = lines[j]
    next unless l =~ /^\s*end\b/ && l[/^\s*/].length == def_indent

    span = lines[def_index..j]
    return span.join if balanced_span?(span)
  end
  nil
end

# Yields { signature:, doc:, body: } for every documented method in a file.
# Any malformed region is skipped; the walk itself never raises.
def documented_methods(rel, code)
  lines = code.lines
  stack = []       # [{ indent:, name: }] — "self" marks `class << self`
  pending_doc = []
  methods = []

  lines.each_with_index do |raw, i|
    line = raw.chomp

    if line =~ /^\s*#/
      pending_doc << line
      next
    end

    doc = pending_doc
    pending_doc = []
    next if line.strip.empty?

    indent = line[/^\s*/].length

    case line
    when SELF_CLASS_RE
      stack.pop while stack.any? && stack.last[:indent] > indent
      stack << { indent: indent, name: "self" }
    when CLASS_RE
      stack.pop while stack.any? && stack.last[:indent] >= indent
      parent = stack.reverse.find { |e| e[:name] != "self" }
      name = parent ? "#{parent[:name]}::#{Regexp.last_match(2)}" : Regexp.last_match(2)
      stack << { indent: indent, name: name }
    when DEF_RE
      stack.pop while stack.any? && stack.last[:indent] >= indent
      next if stack.empty?

      name = Regexp.last_match(1)
      params = Regexp.last_match(2).to_s
      self_method = line =~ DEF_SELF_RE || stack.last[:name] == "self"
      owner = if stack.last[:name] == "self"
                stack.reverse.find { |e| e[:name] != "self" }&.dig(:name)
              else
                stack.last[:name]
              end
      next if owner.nil?

      body = capture_def_body(lines, i, indent, line)
      next if body.nil?

      begin
        doc_lines = clean_doc(doc)
      rescue StandardError
        next
      end
      next if doc_lines.empty?
      doc_text = doc_lines.join("\n")
      next if doc_text.length > 1500

      sep = self_method ? "." : "#"
      methods << { signature: "#{owner}#{sep}#{name}#{params}", doc: doc_text, body: body }
    end
  end

  methods
end

METHOD_PROMPTS = [
  "Implement the Ruby method `%s` (file `%s`) according to this documentation:\n\n%s\n\nReturn only the complete method definition.",
  "A Ruby method in the file `%s` is documented as follows:\n\n%s\n\nWrite the implementation of the method `%s`. Return only the method source.",
  "Here is the API documentation for the Ruby method `%s` from `%s`:\n\n%s\n\nImplement it. Return only the method source.",
].freeze

def build_method_entries(sections, counter)
  entries = []
  sections.each do |rel, code|
    next if test_file?(rel)

    methods = begin
      documented_methods(rel, code)
    rescue StandardError => e
      puts "  [warn] skipping #{rel} (method parsing): #{e.class}: #{e.message}"
      next
    end

    methods.each do |m|
      human = case counter % METHOD_PROMPTS.size
              when 0 then format(METHOD_PROMPTS[0], m[:signature], rel, m[:doc])
              when 1 then format(METHOD_PROMPTS[1], rel, m[:doc], m[:signature])
              else        format(METHOD_PROMPTS[2], m[:signature], rel, m[:doc])
              end
      counter += 1
      entries.concat(split_long_entries(human, m[:body]))
    end
  end
  dedupe_by_answer(entries)
end

# ---------------------------------------------------------------------------
# 3. Test → implementation pairs
# ---------------------------------------------------------------------------

TEST_IMPL_PROMPTS = [
  "Here is the test suite for `%s`:\n\n```ruby\n%s\n```\n\nImplement `%s` so it satisfies these tests. Return only the implementation.",
  "The tests below target the file `%s`. Write the implementation that passes them.\n\n```ruby\n%s\n```",
  "Given these tests for `%s`, implement the class under test:\n\n```ruby\n%s\n```\n\nReturn only the Ruby code.",
].freeze

# <repo>/test/models/room_test.rb → <repo>/lib/models/room.rb or <repo>/app/models/room.rb
# <repo>/spec/models/room_spec.rb → same (RSpec naming).
def implementation_candidates(test_rel)
  base = test_rel.sub(/_(?:test|spec)\.rb\z/, "")
  dir = File.dirname(base)
  name = File.basename(base)

  %w[lib app].map do |root|
    new_dir = dir.sub(%r{(^|/)(?:test|spec)(/|$)}, "\\1#{root}\\2")
    "#{new_dir}/#{name}.rb"
  end
end

def build_test_entries(sections, counter)
  entries = []
  sections.each do |rel, code|
    next unless test_file?(rel)
    next if code.length > 15_000

    begin
      candidates = implementation_candidates(rel)
      target = candidates.find { |c| sections[c] }
      next if target.nil?
      impl = sections[target]
      next if impl.length > 15_000

      human = case counter % TEST_IMPL_PROMPTS.size
              when 0 then format(TEST_IMPL_PROMPTS[0], target, code, target)
              when 1 then format(TEST_IMPL_PROMPTS[1], target, code)
              else        format(TEST_IMPL_PROMPTS[2], target, code)
              end
      counter += 1
      entries.concat(split_long_entries(human, impl))
    rescue StandardError => e
      puts "  [warn] skipping test pair for #{rel}: #{e.class}: #{e.message}"
      next
    end
  end
  dedupe_by_answer(entries)
end

# ---------------------------------------------------------------------------
# 4. Bug-fix pairs mined from git history
# ---------------------------------------------------------------------------

FIX_KEYWORDS = /\b(fix(es|ed|ing)?|bug|correct(ed|ion)?|prevent|handle|avoid|error|crash|invalid|nil|empty|race|deadlock|leak|rescue|regression)\b/i
MAX_BUG_FILE_LINES = 200
# Flat per-repo cap on how many recent commits are scanned for fixes.
MAX_COMMITS_PER_REPO = 2000
MAX_BUG_CANDIDATES = 400

BUG_PROMPTS = [
  "The Ruby code below has a bug: \"%s\".\n\n```ruby\n%s\n```\n\nFix it and return the corrected code.",
  "A bug in this Ruby file was fixed with the note \"%s\":\n\n```ruby\n%s\n```\n\nReturn the corrected version.",
].freeze

def git_show(repo_root, rev, path)
  out, status = Open3.capture2("git", "--no-optional-locks", "-C", repo_root, "show", "#{rev}:#{path}")
  status.success? ? out : nil
rescue StandardError
  nil
end

# Returns [human, answer] pairs for real single-file fixes in a repo's history.
# Everything unexpected (git errors, odd commits, non-UTF-8 output) is skipped.
def bug_fix_pairs_for(repo, repo_root)
  begin
    log, = Open3.capture2("git", "--no-optional-locks", "-C", repo_root,
                          "log", "-n", MAX_COMMITS_PER_REPO.to_s,
                          "--format=%H%x09%s", "--name-status")
  rescue StandardError => e
    puts "  [warn] git log failed for #{repo}: #{e.class}: #{e.message}"
    return []
  end

  pairs = []
  candidates = 0
  commit = nil
  modified = []

  process = lambda do
    next if commit.nil? || modified.size != 1
    subject = commit.split("\t", 2)[1].to_s.force_encoding("UTF-8").scrub
    next unless subject =~ FIX_KEYWORDS

    begin
      sha = commit.split("\t").first
      before = git_show(repo_root, "#{sha}^", modified.first)
      after  = git_show(repo_root, sha, modified.first)
      next if before.nil? || after.nil? || before == after
      before = before.force_encoding("UTF-8").scrub
      after  = after.force_encoding("UTF-8").scrub
      next if before.lines.size > MAX_BUG_FILE_LINES || after.lines.size > MAX_BUG_FILE_LINES

      candidates += 1
      return if candidates > MAX_BUG_CANDIDATES

      human = case pairs.size % BUG_PROMPTS.size
              when 0 then format(BUG_PROMPTS[0], subject, before)
              else        format(BUG_PROMPTS[1], subject, before)
              end
      pairs.concat(split_long_entries(human, after))
    rescue StandardError => e
      puts "  [warn] skipping bug-fix candidate in #{repo}: #{e.class}: #{e.message}"
    end
  end

  log.each_line do |l|
    l = l.chomp
    if l =~ /\A[0-9a-f]{40}\t/
      process.call
      commit = l
      modified = []
    elsif l =~ /\AM\t(.+\.rb)\z/
      modified << Regexp.last_match(1)
    elsif l.empty? && !modified.empty?
      # `--name-status` puts a blank line between the commit header and its
      # status lines, so only a blank AFTER status lines ends the commit.
      process.call
      commit = nil
    end
  end
  process.call

  dedupe_by_answer(pairs)
end

def build_bug_entries(sources_root)
  pairs = []
  repos = Dir.glob(File.join(sources_root, "*"))
             .select { |p| File.directory?(p) }
             .reject { |p| File.basename(p).start_with?(".") }
             .sort

  repos.each do |repo_root|
    repo = File.basename(repo_root)
    puts "  Mining #{repo} git history for bug fixes..."
    pairs.concat(bug_fix_pairs_for(repo, repo_root))
  end
  pairs
end

# ---------------------------------------------------------------------------
# 5. Guide how-to pairs with diversified prompts
# ---------------------------------------------------------------------------

GUIDE_OVERVIEW_PROMPTS = [
  "What does the guide \"%s\" cover? Give an overview.",
  "Summarize the guide \"%s\".",
  "I want an overview of the guide \"%s\". What will I learn?",
  "Give me the highlights of the guide \"%s\".",
  "What topics does the guide \"%s\" walk through?",
].freeze

GUIDE_CHAPTER_PROMPTS = [
  "Explain \"%s\" from the guide \"%s\".",
  "Describe \"%s\" as covered in the guide \"%s\".",
  "Walk me through \"%s\" from the guide \"%s\".",
  "What does the guide \"%s\" say about \"%s\"?",
  "I'm reading the guide \"%s\". Tell me about \"%s\".",
  "Summarize the section \"%s\" of the guide \"%s\".",
  "How does the guide \"%s\" cover \"%s\"?",
  "Teach me \"%s\" from the guide \"%s\".",
  "Can you detail \"%s\" as explained in the guide \"%s\"?",
  "Give me the key points of \"%s\" from the guide \"%s\".",
  "What should I know about \"%s\" according to the guide \"%s\"?",
  "Break down \"%s\" from the guide \"%s\".",
].freeze

def build_guide_entries(counter)
  entries = []
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

    intro = sections.empty? ? guide[:body] : sections.first[:content]
    answer = [guide[:description].to_s.strip, intro.strip].reject(&:empty?).join("\n\n")
    unless answer.empty?
      human = format(GUIDE_OVERVIEW_PROMPTS[counter % GUIDE_OVERVIEW_PROMPTS.size], guide[:title])
      counter += 1
      entries.concat(split_long_entries(human, answer))
    end

    sections.drop(1).each do |section|
      heading = section[:heading].to_s.strip
      content = section[:content].strip
      next if heading.empty? || content.empty?

      human = format(GUIDE_CHAPTER_PROMPTS[counter % GUIDE_CHAPTER_PROMPTS.size], heading, guide[:title])
      counter += 1
      entries.concat(split_long_entries(human, "## #{heading}\n\n#{content}"))
    end
  end

  puts "  Guides skipped (unparseable): #{skipped}" if skipped.positive?
  entries
end

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def dedupe_by_answer(entries)
  seen = {}
  entries.reject do |_h, g|
    key = Digest::SHA256.hexdigest(g)
    seen[key] ? true : (seen[key] = true; false)
  end
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
  puts "  Entries skipped at write time: #{skipped}" if skipped.positive?
end

def validate(path)
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
  sections = {}
  Dir.glob(File.join(CODE_DIR, "*.md")).sort
     .reject { |f| File.basename(f) == "INDEX.md" }
     .each do |md|
    begin
      parse_sections(File.read(md, encoding: "UTF-8")).each do |rel, code|
        sections[rel] = code unless code.strip.empty?
      end
    rescue StandardError => e
      puts "  [warn] skipping #{File.basename(md)}: #{e.class}: #{e.message}"
    end
  end
  puts "#{sections.size} source files loaded from code_context/"

  counter = 0
  entries = []
  counts = {}

  base = load_base_entries(DATASET_DIR)
  entries.concat(base)
  counts[:base] = base.size
  puts "  base entries (deduplicated): #{base.size}"

  method_entries = build_method_entries(sections, counter)
  counter += method_entries.size
  entries.concat(method_entries)
  counts[:method_impl] = method_entries.size
  puts "  method-implementation pairs: #{method_entries.size}"

  test_entries = build_test_entries(sections, counter)
  counter += test_entries.size
  entries.concat(test_entries)
  counts[:test_impl] = test_entries.size
  puts "  test → implementation pairs: #{test_entries.size}"

  bug_entries = []
  unless SKIP_BUGS
    bug_entries = build_bug_entries(SOURCES_ROOT)
    counter += bug_entries.size
    entries.concat(bug_entries)
    counts[:bug_fix] = bug_entries.size
    puts "  bug-fix pairs: #{bug_entries.size}"
  end

  guide_entries = build_guide_entries(counter)
  entries.concat(guide_entries)
  counts[:guide_howto] = guide_entries.size
  puts "  guide how-to pairs (diversified prompts): #{guide_entries.size}"

  FileUtils.mkdir_p(File.dirname(OUTPUT))
  write_jsonl(OUTPUT, entries)

  total = validate(OUTPUT)
  bytes = File.size(OUTPUT)
  puts
  puts "Wrote #{total} entries (#{format('%.1f', bytes / 1024.0)} KB, ~#{bytes / 4} tokens) to #{OUTPUT}"
  puts counts.map { |k, v| "#{k}=#{v}" }.join(", ")
end

main if __FILE__ == $PROGRAM_NAME
