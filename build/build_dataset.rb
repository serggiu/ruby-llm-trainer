#!/usr/bin/env ruby
# frozen_string_literal: true

# Data root — where _sources/, _dataset/, etc. live. Override with
# LLM_TRAINER_ROOT (used by tests to run the pipeline in a sandbox).
ROOT = ENV["LLM_TRAINER_ROOT"] || File.expand_path("..", File.dirname(__FILE__))

# Build a fine-tuning dataset from a code-context Markdown dump.
#
# Reads a code-context Markdown dump (as produced by build_code_context.rb)
# and emits a JSONL dataset in the ShareGPT conversation format used by local
# fine-tuning pipelines such as LLaMA-Factory, axolotl, and Unsloth:
#
#   {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
#
# Entry types generated per source file:
#   - code reproduction  — the full, verbatim Ruby source (one per file)
#   - API explanation    — purpose + public API, assembled from the real RDoc
#                          comments and declarations in the file (lib files only)
#   - test coverage      — the real `test "..."` descriptions (test files only)
#
# Every entry is capped at MAX_CONTEXT_TOKENS (2048 estimated tokens):
# answers longer than the cap are split into as many "(part i/n)" pairs as
# needed, so no generated pair exceeds the cap.
#
# Usage:
#   ruby build/build_dataset.rb [input.md] [output.jsonl]
#
# Defaults: the first dump in code_context/ → _dataset/<dump>.jsonl

require "fileutils"
require "json"

DEFAULT_MD = Dir.glob(File.join(ROOT, "code_context", "*.md"))
                    .reject { |f| File.basename(f) == "INDEX.md" }
                    .sort
                    .first
INPUT_MD = File.expand_path(ARGV[0] || DEFAULT_MD.to_s)
OUTPUT_JSONL = File.expand_path(
  ARGV[1] || if DEFAULT_MD
               File.join(ROOT, "_dataset", "#{File.basename(DEFAULT_MD, ".md")}.jsonl")
             else
               File.join(ROOT, "_dataset", "dataset.jsonl")
             end
)

# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

# Splits the code-context markdown into [rel_path, code] pairs, one per file.
def parse_sections(md)
  sections = []
  md.split(/^<a id="/).drop(1).each do |part|
    path = part[%r{^[^"]*"></a>\n\n## `([^`]+)`}, 1]
    code = part[/^```ruby\n(.*?)\n^```$/m, 1]
    sections << [path, code] if path && code
  end
  sections
end

# ---------------------------------------------------------------------------
# Ruby API extraction (regex-based, tolerant of unusual formatting)
# ---------------------------------------------------------------------------

CLASS_DECL    = /^\s*(class|module)\s+([A-Z]\w*)\b.*$/
SELF_CLASS    = /^\s*class\s*<<\s*self\b/
DEF_DECL      = /^\s*def\s+(?:self\.)?([a-zA-Z_]\w*[!?=]?)(\s*\([^;]*\))?.*$/
DEF_SELF      = /^\s*def\s+self\./
INCLUDE_DECL  = /^\s*(?:include|extend|prepend)\s+([A-Z][\w:]*)/
TEST_DECL     = /^\s*test\s+["']([^"']+)["']/

# Returns { file_doc:, classes: [{ name:, kind:, doc:, includes: [], class_methods: [], instance_methods: [] }] }
# for the given Ruby source. Docs are the real comment blocks directly above
# each declaration; `:nodoc:` markers and license banners are filtered out.
def extract_api(code)
  api = { file_doc: nil, classes: [] }
  stack = []   # [{ indent:, entry: }]
  pending_doc = []

  code.each_line do |raw|
    line = raw.chomp

    if line =~ /^\s*#/
      pending_doc << line
      next
    end

    doc = pending_doc
    pending_doc = []
    next if line.strip.empty? # a blank line detaches the comment run

    indent = line[/^\s*/].length

    case line
    when SELF_CLASS
      stack.pop while stack.any? && stack.last[:indent] > indent
      stack << { indent: indent, entry: nil, self_class: true }
    when CLASS_DECL
      stack.pop while stack.any? && stack.last[:indent] >= indent
      parent = stack.reverse.find { |f| !f[:self_class] }
      parent = parent && parent[:entry]
      name = parent ? "#{parent[:name]}::#{Regexp.last_match(2)}" : Regexp.last_match(2)
      entry = {
        name: name, kind: Regexp.last_match(1),
        doc: clean_doc(doc), includes: [],
        class_methods: [], instance_methods: [],
      }
      stack << { indent: indent, entry: entry }
      api[:classes] << entry
      if api[:file_doc].nil? && !entry[:doc].empty?
        api[:file_doc] = entry[:doc]
      end
    when DEF_DECL
      stack.pop while stack.any? && stack.last[:indent] >= indent
      top = stack.last
      next if top.nil? # top-level def, not part of a class/module body

      name     = Regexp.last_match(1)
      params   = Regexp.last_match(2).to_s
      self_method = line =~ DEF_SELF || top[:self_class]
      owner = top[:self_class] ? stack.reverse.find { |f| !f[:self_class] }&.dig(:entry) : top[:entry]
      next if owner.nil?

      method_doc = clean_doc(doc)
      method = { name: name, params: params, doc: method_doc }

      if self_method
        owner[:class_methods] << method
      else
        owner[:instance_methods] << method
      end
    when INCLUDE_DECL
      top = stack.reverse.find { |f| !f[:self_class] }
      top && top[:entry][:includes] << Regexp.last_match(1)
    end
  end

  api
end

# Converts a raw comment run (e.g. "    # some text") into cleaned doc lines.
# Indentation before "#" is stripped but everything after "# " is preserved,
# so RDoc code examples keep their relative indentation.
def clean_doc(run)
  return [] if run.empty?
  return [] if run.any? { |l| l =~ /^\s*#--/ || l.include?("Copyright (c)") }

  lines = run.map { |l| l.sub(/^\s*#\s?/, "") }
  lines.reject! { |l| l =~ /^\s*:nodoc:\s*$/ }
  lines.shift while lines.first && lines.first.strip.empty?
  lines.pop while lines.last && lines.last.strip.empty?
  lines
end

def doc_text(lines)
  lines.join("\n")
end

# First paragraph of a doc block — used to keep API summaries compact.
def doc_summary(lines)
  lines.take_while { |l| !l.strip.empty? }.join(" ")
end

# ---------------------------------------------------------------------------
# Entry builders
# ---------------------------------------------------------------------------

CODE_PROMPTS = [
  "Show me the complete Ruby source code of `%s`.",
  "Write the full implementation of `%s`.",
  "Reproduce the entire source of `%s` verbatim.",
  "What does the file `%s` contain? Output its full Ruby source code.",
].freeze

EXPLAIN_PROMPTS = [
  "Explain what the file `%s` does and describe the API it defines.",
  "Walk me through the purpose of `%s` and its public methods.",
  "What is the role of `%s`? Describe the classes, modules, and methods it contributes.",
].freeze

TEST_PROMPTS = [
  "Show me the test suite for `%s`.",
  "How is `%s` tested? Reproduce the test file in full.",
  "Output the complete test code of `%s`.",
].freeze

COVERAGE_PROMPTS = [
  "What behaviors does the test file `%s` cover? List all the test cases.",
  "What is tested in `%s`? Enumerate the covered behaviors.",
  "Read `%s` and list every scenario its tests exercise.",
].freeze

# Builds the "API explanation" answer for a lib file from its real contents.
def build_explanation(path, code)
  api = extract_api(code)
  return nil if api[:classes].empty?

  lines = []
  lines << "The file `#{path}` is Ruby source code. Here is its purpose and public API."
  lines << ""

  unless api[:file_doc].nil? || api[:file_doc].empty?
    lines << "## Purpose"
    lines << ""
    lines << doc_text(api[:file_doc])
    lines << ""
  end

  lines << "## API"
  lines << ""

  api[:classes].each do |cls|
    has_children = !cls[:includes].empty? ||
      !cls[:class_methods].empty? || !cls[:instance_methods].empty? ||
      (!cls[:doc].empty? && cls[:doc] != api[:file_doc])

    next unless has_children

    lines << "- `#{cls[:name]}` (#{cls[:kind]})"
    if !cls[:doc].empty? && cls[:doc] != api[:file_doc]
      lines << "  - Purpose: #{doc_summary(cls[:doc])}"
    end
    unless cls[:includes].empty?
      lines << "  - includes: #{cls[:includes].map { |i| "`#{i}`" }.join(", ")}"
    end
    unless cls[:class_methods].empty?
      lines << "  - class methods:"
      cls[:class_methods].each do |m|
        lines << "    - `.#{m[:name]}#{m[:params]}`" + (m[:doc].empty? ? "" : " — #{doc_summary(m[:doc])}")
      end
    end
    unless cls[:instance_methods].empty?
      lines << "  - instance methods:"
      cls[:instance_methods].each do |m|
        lines << "    - `##{m[:name]}#{m[:params]}`" + (m[:doc].empty? ? "" : " — #{doc_summary(m[:doc])}")
      end
    end
  end

  lines.join("\n")
end

# Builds the "test coverage" answer from the real `test "..."` descriptions.
def build_coverage(path, code)
  tests = code.scan(TEST_DECL).flatten
  return nil if tests.empty?

  lines = []
  lines << "The test file `#{path}` exercises the following behaviors:"
  lines << ""
  tests.each { |t| lines << "- #{t}" }
  lines.join("\n")
end


# ---------------------------------------------------------------------------
# Context-length bounding: entries longer than MAX_CONTEXT_TOKENS are split
# into as many pairs as needed so that no generated pair exceeds the cap.
# ---------------------------------------------------------------------------

# Hard cap on the estimated token length of any generated conversation pair.
# 2048 keeps single-sample memory at ~14 GB peak for an 8B 4-bit model
# (mlx-lm's quantized attention materializes O(n^2) attention scores), which
# fits a 32 GB Mac with room to spare. Tunable — lower it for smaller
# machines, raise it only if you know the memory math.
MAX_CONTEXT_TOKENS = 2048

# Chat-template overhead (system/role/turn special tokens) added by the
# tokenizer on top of the raw prompt + answer text.
CHAT_TEMPLATE_OVERHEAD_TOKENS = 64

# Conservative chars-per-token factor used when deciding and sizing splits:
# derived from the measured corpus floor of 1.62 chars/token (base64-encoded
# CBOR in a WebAuthn test file); 1.4 keeps ~14% margin below that floor, so
# generated pairs stay safely under MAX_CONTEXT_TOKENS with the real
# tokenizer. Typical code is ~4.4 chars/token — the factor only bites on
# pathological dense content.
def split_chars_per_token
  1.4
end

# Rough token estimate used for context bounding (see split_chars_per_token).
def estimate_tokens(text)
  (text.bytesize / split_chars_per_token).ceil
end

# Greedy line-based packing of +text+ into chunks of at most +max_tokens+
# estimated tokens each. Lines are kept whole; a single over-long line
# (minified code, ...) is split by characters. Chunks are measured in BYTES
# (estimate_tokens counts bytes), so multibyte content (emoji, CJK) never
# lets a chunk exceed the token budget.
def split_text_chunks(text, max_tokens)
  return [] if text.empty?

  max_bytes = (max_tokens * split_chars_per_token).floor
  chunks = []
  current = +""
  current_bytes = 0

  text.lines.each do |line|
    line_bytes = line.bytesize

    if line_bytes > max_bytes
      # Single over-long line: flush the pending chunk, then split the line
      # by characters into byte-budgeted pieces (multibyte-safe).
      chunks << current unless current.empty?
      piece = +""
      piece_bytes = 0
      line.each_char do |ch|
        if !piece.empty? && piece_bytes + ch.bytesize > max_bytes
          chunks << piece
          piece = +""
          piece_bytes = 0
        end
        piece << ch
        piece_bytes += ch.bytesize
      end
      chunks << piece unless piece.empty?
      current = +""
      current_bytes = 0
    else
      if !current.empty? && current_bytes + line_bytes > max_bytes
        chunks << current
        current = +""
        current_bytes = 0
      end
      current << line
      current_bytes += line_bytes
    end
  end

  chunks << current unless current.empty?
  chunks
end

# Returns [[human, answer], ...] — the pair as-is when it fits under
# MAX_CONTEXT_TOKENS (estimated, including chat-template overhead), otherwise
# as many pairs as needed, each under the cap:
#   - answer too long (the common case: verbatim code, long guide chapters):
#     the answer is split; the prompt is repeated with a "part i/n" suffix
#   - prompt too long (e.g. a huge test file as prompt): the prompt is split
#     and each part is paired with the answer
#   - both too long: both are split and paired part-by-part
# Splitting is line-based, so code and markdown stay readable in each part.
def split_long_entries(human, answer)
  return [[human, answer]] if human.empty? || answer.empty?

  if estimate_tokens("#{human}\n#{answer}") + CHAT_TEMPLATE_OVERHEAD_TOKENS <= MAX_CONTEXT_TOKENS
    return [[human, answer]]
  end

  human_tokens = estimate_tokens(human)
  answer_tokens = estimate_tokens(answer)

  if answer_tokens <= MAX_CONTEXT_TOKENS - CHAT_TEMPLATE_OVERHEAD_TOKENS - 32
    # Answer fits beside a chunked prompt: split the prompt to fit.
    prompt_budget = MAX_CONTEXT_TOKENS - CHAT_TEMPLATE_OVERHEAD_TOKENS - answer_tokens - 16
    parts = split_text_chunks(human, prompt_budget)
    return [[human, answer]] if parts.empty?

    n = parts.size
    parts.each_with_index.map { |hp, i| ["#{hp} (part #{i + 1}/#{n})", answer] }
  elsif human_tokens <= MAX_CONTEXT_TOKENS - CHAT_TEMPLATE_OVERHEAD_TOKENS - 32
    # Prompt fits beside a chunked answer: split the answer.
    answer_budget = MAX_CONTEXT_TOKENS - CHAT_TEMPLATE_OVERHEAD_TOKENS - human_tokens - 16
    parts = split_text_chunks(answer, answer_budget)
    return [[human, answer]] if parts.empty?

    n = parts.size
    parts.each_with_index.map { |ap, i| ["#{human} (part #{i + 1}/#{n})", ap] }
  else
    # Both long: split both and pair part-by-part.
    budget = (MAX_CONTEXT_TOKENS - CHAT_TEMPLATE_OVERHEAD_TOKENS) / 2 - 16
    human_parts = split_text_chunks(human, budget)
    answer_parts = split_text_chunks(answer, budget)
    return [[human, answer]] if human_parts.empty? || answer_parts.empty?

    n = [human_parts.size, answer_parts.size].max
    (0...n).map do |i|
      hp = human_parts[[i, human_parts.size - 1].min]
      ap = answer_parts[[i, answer_parts.size - 1].min]
      ["#{hp} (part #{i + 1}/#{n})", ap]
    end
  end
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main
  abort "Input not found: #{INPUT_MD}" unless File.file?(INPUT_MD)

  sections = parse_sections(File.read(INPUT_MD, encoding: "UTF-8"))
  abort "No file sections found in #{INPUT_MD}" if sections.empty?

  entries = []
  counter = 0

  sections.each do |path, code|
    is_test = path.include?("/test/")

    # 1. Code reproduction entry
    prompt_pool = is_test ? TEST_PROMPTS : CODE_PROMPTS
    human = format(prompt_pool[counter % prompt_pool.size], path)
    counter += 1
    entries.concat(split_long_entries(human, code))

    # 2. API explanation entry (lib files only)
    unless is_test
      explanation = build_explanation(path, code)
      if explanation
        human = format(EXPLAIN_PROMPTS[counter % EXPLAIN_PROMPTS.size], path)
        counter += 1
        entries.concat(split_long_entries(human, explanation))
      end
    end

    # 3. Test coverage entry (test files only)
    if is_test
      coverage = build_coverage(path, code)
      if coverage
        human = format(COVERAGE_PROMPTS[counter % COVERAGE_PROMPTS.size], path)
        counter += 1
        entries.concat(split_long_entries(human, coverage))
      end
    end
  end

  FileUtils.mkdir_p(File.dirname(OUTPUT_JSONL))

  File.open(OUTPUT_JSONL, "w:UTF-8") do |f|
    entries.each do |human, gpt|
      f.puts JSON.generate(
        "conversations" => [
          { "from" => "human", "value" => human },
          { "from" => "gpt",   "value" => gpt },
        ],
      )
    end
  end

  bytes = File.size(OUTPUT_JSONL)
  chars = entries.sum { |h, g| h.length + g.length }
  puts "Parsed #{sections.size} file sections from #{File.basename(INPUT_MD)}"
  puts "Generated #{entries.size} conversation entries"
  puts "Wrote #{format('%.1f', bytes / 1024.0)} KB (~#{chars / 4} tokens) to #{OUTPUT_JSONL}"
end

main if __FILE__ == $PROGRAM_NAME
