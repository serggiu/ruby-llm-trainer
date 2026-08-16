#!/usr/bin/env ruby
# frozen_string_literal: true

# Tests for the context-length bounding logic in build_dataset.rb
# (MAX_CONTEXT_TOKENS, split_text_chunks, split_long_entries).

require_relative "helper"
require_relative "../build_dataset"

def build_long_answer
  (1..900).map { |i| "def method_#{i}\n  # doc for #{i}\n  #{i} * 2\nend\n" }.join
end

def build_long_prompt
  (1..600).map { |i| "test_#{i} \"x\" do\n  assert true\nend\n" }.join
end

def max_pair_tokens(entries)
  entries.map { |h, a| estimate_tokens(h) + estimate_tokens(a) }.max
end

cap = MAX_CONTEXT_TOKENS - CHAT_TEMPLATE_OVERHEAD_TOKENS

# --- 1. Long answer, short prompt (the common case) ---
long_answer = build_long_answer
entries = split_long_entries("Show me the complete Ruby source code of `big.rb`.", long_answer)
assert entries.size > 1, "long answer is split into multiple pairs"
assert max_pair_tokens(entries) <= cap, "every pair fits under the cap"
assert entries.all? { |h, _| h.include?("(part ") }, "prompts annotated with part numbers"
assert_equal long_answer, entries.map { |_, a| a }.join, "no content lost or duplicated"

# --- 2. Long prompt, short answer (e.g. huge test file as prompt) ---
long_prompt = build_long_prompt
short_answer = "class Thing\nend\n"
entries2 = split_long_entries(long_prompt, short_answer)
assert entries2.size > 1, "long prompt is split into multiple pairs"
assert max_pair_tokens(entries2) <= cap, "every pair fits under the cap"
assert entries2.all? { |_, a| a == short_answer }, "answer repeated per prompt part"

# --- 3. Both long: split both, paired part-by-part ---
entries3 = split_long_entries(build_long_prompt, build_long_answer)
assert entries3.size > 1, "both-long case produces multiple pairs"
assert max_pair_tokens(entries3) <= cap, "every pair fits under the cap"

# --- 4. Short pair stays untouched ---
assert_equal [["short prompt", "short answer"]],
             split_long_entries("short prompt", "short answer"),
             "short pair passes through unchanged"

# --- 5. Single over-long line (hard split, must reassemble exactly) ---
giant = "x" * 100_000
entries4 = split_long_entries("p", giant)
assert entries4.size > 1, "giant line is hard-split"
assert max_pair_tokens(entries4) <= cap, "every pair fits under the cap"
assert_equal giant, entries4.map { |_, a| a }.join, "giant line reassembles exactly"

# --- 6. Empty sides are returned as-is (never dropped) ---
assert_equal [["", "answer"]], split_long_entries("", "answer"), "empty prompt kept"
assert_equal [["prompt", ""]], split_long_entries("prompt", ""), "empty answer kept"

puts "  (#{entries.size} + #{entries2.size} + #{entries3.size} + #{entries4.size} pairs generated across split cases)"
