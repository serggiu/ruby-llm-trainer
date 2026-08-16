#!/usr/bin/env ruby
# frozen_string_literal: true

# Measure the token-length distribution of a dataset (diagnostic helper).
#
# Reports sample count, min/mean/p50/p90/p99/max lengths, and how many
# samples exceed max_len — i.e. how much of the data gets truncated during
# training.
#
# Two counting modes:
#   - with a model directory: the real tokenizer (tools/tokenizer.rb — byte-
#     level BPE + chat template, verified identical to the original
#     Python/transformers tokenizer on this pipeline's data)
#   - without: the generator's conservative estimator (SPLIT_CHARS_PER_TOKEN
#     in build_dataset.rb, ~1.4 chars/byte + chat-template overhead)
#
# Accepts both ShareGPT {"conversations": [...]} and MLX {"messages": [...]}.
#
# Usage:
#   ruby tools/seqlen_stats.rb dataset.jsonl [max_len]
#   ruby tools/seqlen_stats.rb dataset.jsonl <model_dir> [max_len]
#   (default max_len = MAX_CONTEXT_TOKENS)

require "json"
require_relative "../build_dataset"
require_relative "tokenizer"

PATH = ARGV[0]
abort "usage: #{$PROGRAM_NAME} dataset.jsonl [model_dir] [max_len]" if PATH.nil? || !File.file?(PATH)

REST = ARGV[1..]
MODEL_DIR = REST.find { |a| File.directory?(a) }
MAX_LEN = REST.find { |a| a =~ /\A\d+\z/ }&.to_i || MAX_CONTEXT_TOKENS

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

lens = []
bad = 0

File.foreach(PATH, encoding: "UTF-8") do |line|
  next if line.strip.empty?

  begin
    msgs = messages_from(JSON.parse(line))
  rescue JSON::ParserError => e
    bad += 1
    puts "  [warn] line #{$.}: #{e.message}"
    next
  end

  lens << entry_length(msgs)
end

lens.sort!
n = lens.size
if n.zero?
  puts "no samples"
  exit 0
end

pct = lambda do |p|
  lens[[(n * p).to_i, n - 1].min]
end

over = lens.count { |l| l > MAX_LEN }
label = TOKENIZER ? "tokens" : "est tokens"
puts "samples: #{n} (skipped #{bad})"
puts format("#{label}: min %d  mean %d  p50 %d  p90 %d  p99 %d  max %d",
            lens.first, lens.sum / n, pct.call(0.5), pct.call(0.9), pct.call(0.99), lens.last)
puts format("> %d #{label} (truncated during training): %d (%.1f%%)",
            MAX_LEN, over, over * 100.0 / n)
puts "longest few: #{lens.last(5).join(', ')}" if over.positive?
