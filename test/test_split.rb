#!/usr/bin/env ruby
# frozen_string_literal: true

# Quick verification of split_long_entries.
require_relative "../build_dataset"

def section(label)
  puts
  puts "== #{label} =="
end

# 1. Long answer, short prompt (the common case)
section "long answer"
long_answer = (1..900).map { |i| "def method_#{i}\n  # doc for #{i}\n  #{i} * 2\nend\n" }.join
entries = split_long_entries("Show me the complete Ruby source code of `big.rb`.", long_answer)
max = entries.map { |h, a| estimate_tokens(h) + estimate_tokens(a) }.max
puts "entries: #{entries.size}, max estimated pair tokens: #{max}, cap: #{MAX_CONTEXT_TOKENS - CHAT_TEMPLATE_OVERHEAD_TOKENS}"
entries.first(3).each_with_index { |(h, _), i| puts "  prompt[#{i}]: #{h[0, 90]}" }
puts "answer chars kept: #{entries.sum { |_, a| a.length }} vs original #{long_answer.length}"

# 2. Long prompt, short answer (huge test file as prompt)
section "long prompt"
long_prompt = (1..600).map { |i| "test_#{i} \"x\" do\n  assert true\nend\n" }.join
short_answer = "class Thing\nend\n"
entries2 = split_long_entries(long_prompt, short_answer)
max2 = entries2.map { |h, a| estimate_tokens(h) + estimate_tokens(a) }.max
puts "entries: #{entries2.size}, max: #{max2}, answer repeated: #{entries2.all? { |_, a| a == short_answer }}"

# 3. Both long
section "both long"
entries3 = split_long_entries(long_prompt, long_answer)
max3 = entries3.map { |h, a| estimate_tokens(h) + estimate_tokens(a) }.max
puts "entries: #{entries3.size}, max: #{max3}"

# 4. Short pair unchanged
section "short pair"
e = split_long_entries("short prompt", "short answer")
puts "unchanged: #{e == [["short prompt", "short answer"]]}"

# 5. Single over-long line (hard char split)
section "single giant line"
giant = ("x" * 100_000)
entries4 = split_long_entries("p", giant)
max4 = entries4.map { |h, a| estimate_tokens(h) + estimate_tokens(a) }.max
puts "entries: #{entries4.size}, max: #{max4}, reassembled: #{entries4.map { |_, a| a }.join == giant}"

puts
puts "ALL GOOD" if [max, max2, max3, max4].all? { |m| m <= MAX_CONTEXT_TOKENS - CHAT_TEMPLATE_OVERHEAD_TOKENS }
