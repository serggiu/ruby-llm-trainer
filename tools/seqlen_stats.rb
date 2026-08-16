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

# Collects the length of every parseable line of `lines` (any enumerable of
# JSONL lines). Returns [lens, skipped_count].
def collect_lengths(lines, tokenizer: nil)
  lens = []
  bad = 0
  lines.each_with_index do |line, i|
    next if line.strip.empty?

    begin
      msgs = messages_from(JSON.parse(line))
    rescue JSON::ParserError => e
      bad += 1
      warn "  [warn] line #{i + 1}: #{e.message}"
      next
    end

    lens << entry_length(msgs, tokenizer)
  end
  [lens, bad]
end

# Computes the distribution stats from a SORTED lens array (must not be
# empty). Returns a hash with samples/min/mean/p50/p90/p99/max/over/
# over_pct/label.
def summarize(lens, max_len, real:)
  n = lens.size
  pct = lambda do |p|
    lens[[(n * p).to_i, n - 1].min]
  end
  over = lens.count { |l| l > max_len }
  {
    samples: n,
    min: lens.first, mean: lens.sum / n,
    p50: pct.call(0.5), p90: pct.call(0.9), p99: pct.call(0.99), max: lens.last,
    over: over, over_pct: over * 100.0 / n,
    label: real ? "tokens" : "est tokens",
  }
end

if __FILE__ == $PROGRAM_NAME
  PATH = ARGV[0]
  abort "usage: #{$PROGRAM_NAME} dataset.jsonl [model_dir] [max_len]" if PATH.nil? || !File.file?(PATH)

  REST = ARGV[1..]
  MODEL_DIR = REST.find { |a| File.directory?(a) }
  MAX_LEN = REST.find { |a| a =~ /\A\d+\z/ }&.to_i || MAX_CONTEXT_TOKENS

  TOKENIZER = MODEL_DIR ? Tokenizer.new(MODEL_DIR) : nil

  lens, bad = collect_lengths(File.foreach(PATH, encoding: "UTF-8"), tokenizer: TOKENIZER)
  lens.sort!
  if lens.empty?
    puts "no samples"
    exit 0
  end

  s = summarize(lens, MAX_LEN, real: !TOKENIZER.nil?)
  puts "samples: #{s[:samples]} (skipped #{bad})"
  puts format("#{s[:label]}: min %d  mean %d  p50 %d  p90 %d  p99 %d  max %d",
              s[:min], s[:mean], s[:p50], s[:p90], s[:p99], s[:max])
  puts format("> %d #{s[:label]} (truncated during training): %d (%.1f%%)",
              MAX_LEN, s[:over], s[:over_pct])
  puts "longest few: #{lens.last(5).join(', ')}" if s[:over].positive?
end
