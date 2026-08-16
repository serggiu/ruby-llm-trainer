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
require_relative "../build_dataset"
require_relative "tokenizer"

PATH = ARGV[0]
abort "usage: #{$PROGRAM_NAME} dataset.jsonl [model_dir] [max_len] [limit]" if PATH.nil? || !File.file?(PATH)

REST = ARGV[1..]
MODEL_DIR = REST.find { |a| File.directory?(a) }
NUMS = REST.select { |a| a =~ /\A\d+\z/ }
MAX_LEN = NUMS[0]&.to_i || MAX_CONTEXT_TOKENS
LIMIT = NUMS[1]&.to_i || 20

TOKENIZER = MODEL_DIR ? Tokenizer.new(MODEL_DIR) : nil

def messages_from(obj)
  if obj["messages"]
    obj["messages"]
  else
    obj["conversations"].map do |c|
      { "role" => c["from"] == "human" ? "user" : "assistant", "content" => c["value"] }
    end
  end
end

def entry_length(msgs)
  return TOKENIZER.count_messages(msgs) if TOKENIZER

  estimate_tokens(msgs.map { |m| m["content"].to_s }.join("\n")) + CHAT_TEMPLATE_OVERHEAD_TOKENS
end

shown = 0
File.foreach(PATH, encoding: "UTF-8") do |line|
  next if line.strip.empty?

  begin
    msgs = messages_from(JSON.parse(line))
  rescue JSON::ParserError => e
    warn "  [warn] line #{$.}: #{e.message}"
    next
  end

  len = entry_length(msgs)
  next unless len > MAX_LEN

  contents = msgs.map { |m| m["content"].to_s }
  chars = contents.sum(&:bytesize)
  prompt = contents.first.to_s.gsub("\n", " ")[0, 120]
  label = TOKENIZER ? "#{len} tokens" : "~#{len} est tokens"
  puts format("line %d: %s, %d chars (c=%.2f) | %s", $., label, chars, chars.to_f / len, prompt)
  shown += 1
  break if shown >= LIMIT
end
puts "(showing first #{shown})"
