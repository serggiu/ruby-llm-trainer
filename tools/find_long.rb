#!/usr/bin/env ruby
# frozen_string_literal: true

# List dataset entries whose token length exceeds a threshold (diagnostic
# helper).
#
# Two counting modes (same as seqlen_stats.rb):
#   - with a model directory: the real tokenizer (tools/tokenizer.rb)
#   - without: the generator's conservative estimator
#
# Accepts both ShareGPT {"conversations": [...]} and MLX {"messages": [...]}.
#
# Usage:
#   ruby tools/find_long.rb dataset.jsonl [max_len] [limit]
#   ruby tools/find_long.rb dataset.jsonl <model_dir> [max_len] [limit]
#   (defaults: max_len = MAX_CONTEXT_TOKENS, limit = 20)

require "json"
require_relative "../build/build_dataset"
require_relative "tokenizer"

def messages_from(obj)
  if obj["messages"]
    obj["messages"]
  else
    obj["conversations"].map do |c|
      { "role" => c["from"] == "human" ? "user" : "assistant", "content" => c["value"] }
    end
  end
end

def entry_length(msgs, tokenizer)
  return tokenizer.count_messages(msgs) if tokenizer

  estimate_tokens(msgs.map { |m| m["content"].to_s }.join("\n")) + CHAT_TEMPLATE_OVERHEAD_TOKENS
end

# Returns [{ line:, len:, chars:, prompt: }] for the first `limit` entries of
# `lines` (any enumerable of JSONL lines) whose length exceeds `max_len`.
def find_over_limit(lines, max_len:, limit:, tokenizer: nil)
  results = []
  lines.each_with_index do |line, i|
    next if line.strip.empty?

    begin
      msgs = messages_from(JSON.parse(line))
    rescue JSON::ParserError => e
      warn "  [warn] line #{i + 1}: #{e.message}"
      next
    end

    len = entry_length(msgs, tokenizer)
    next unless len > max_len

    contents = msgs.map { |m| m["content"].to_s }
    chars = contents.sum(&:bytesize)
    prompt = contents.first.to_s.gsub("\n", " ")[0, 120]
    results << { line: i + 1, len: len, chars: chars, prompt: prompt }
    break if results.size >= limit
  end
  results
end

if __FILE__ == $PROGRAM_NAME
  PATH = ARGV[0]
  abort "usage: #{$PROGRAM_NAME} dataset.jsonl [model_dir] [max_len] [limit]" if PATH.nil? || !File.file?(PATH)

  REST = ARGV[1..]
  MODEL_DIR = REST.find { |a| File.directory?(a) }
  NUMS = REST.select { |a| a =~ /\A\d+\z/ }
  MAX_LEN = NUMS[0]&.to_i || MAX_CONTEXT_TOKENS
  LIMIT = NUMS[1]&.to_i || 20

  TOKENIZER = MODEL_DIR ? Tokenizer.new(MODEL_DIR) : nil

  results = find_over_limit(File.foreach(PATH, encoding: "UTF-8"),
                            max_len: MAX_LEN, limit: LIMIT, tokenizer: TOKENIZER)
  results.each do |r|
    label = TOKENIZER ? "#{r[:len]} tokens" : "~#{r[:len]} est tokens"
    puts format("line %d: %s, %d chars (c=%.2f) | %s",
                r[:line], label, r[:chars], r[:chars].to_f / r[:len], r[:prompt])
  end
  puts "(showing first #{results.size})"
end
