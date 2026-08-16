#!/usr/bin/env ruby
# frozen_string_literal: true

# Convert the ShareGPT-format SFT set into MLX training data and slice it.
#
# MLX (mlx_lm.lora) expects one of:
#   {"text": ...}                          — plain text (continued pretraining)
#   {"messages": [{"role": ..., ...}]}     — chat format (SFT)
#
# This script converts _dataset/sft_train_set.jsonl (ShareGPT conversations)
# to chat format and writes _mlx/train.jsonl + _mlx/valid.jsonl. The default
# slice (300 train / 50 valid) is sized for a smoke run; pass sizes as args
# for a full run. (The CPT corpus _pretrain/ruby_corpus.jsonl is already in
# the {"text": ...} format MLX accepts, so it needs no conversion.)
#
# Usage:
#   ruby build_mlx_data.rb                 # → _mlx/{train,valid}.jsonl (300/50)
#   ruby build_mlx_data.rb 12000 1000      # bigger slice
#   ruby build_mlx_data.rb 0 0 --full      # whole set: 0 = "no limit"

require "fileutils"
require "json"

POS_ARGS   = ARGV.reject { |a| a.start_with?("--") }
FLAGS      = ARGV.select { |a| a.start_with?("--") }

TRAIN_COUNT = Integer(POS_ARGS[0] || 300)
VALID_COUNT = Integer(POS_ARGS[1] || 50)
INPUT       = File.expand_path(POS_ARGS[2] || File.join(File.dirname(__FILE__), "_dataset", "sft_train_set.jsonl"))
OUT_DIR     = File.expand_path(POS_ARGS[3] || File.join(File.dirname(__FILE__), "_mlx"))
# Entries longer than this many characters are dropped (0 = keep everything).
# Extremely long entries (100k+ chars) tokenize to >100k tokens and have been
# observed to destabilize training; ~80k chars ≈ 20k tokens.
MAX_CHARS = begin
  flag = FLAGS.find { |a| a.start_with?("--max-chars=") }
  flag ? Integer(flag.split("=", 2)[1]) : 0
end

def to_mlx_line(human, gpt)
  JSON.generate(
    "messages" => [
      { "role" => "user",      "content" => human },
      { "role" => "assistant", "content" => gpt },
    ],
  )
end

def limited?(count, limit)
  !limit.zero? && count >= limit
end

FileUtils.mkdir_p(OUT_DIR)

train = File.open(File.join(OUT_DIR, "train.jsonl"), "w:UTF-8")
valid = File.open(File.join(OUT_DIR, "valid.jsonl"), "w:UTF-8")

train_count = 0
valid_count = 0
skipped = 0
skipped_long = 0

File.foreach(INPUT) do |line|
  begin
    convs = JSON.parse(line)["conversations"]
    human = convs[0]["value"]
    gpt   = convs[1]["value"]
    raise "bad entry shape" unless human.is_a?(String) && gpt.is_a?(String)
  rescue StandardError => e
    skipped += 1
    puts "  [warn] skipping entry: #{e.message}"
    next
  end

  if !MAX_CHARS.zero? && (human.length + gpt.length) > MAX_CHARS
    skipped_long += 1
    next
  end

  out = if !limited?(train_count, TRAIN_COUNT)
          train
        elsif !limited?(valid_count, VALID_COUNT)
          valid
        else
          break
        end

  out.puts to_mlx_line(human, gpt)
  out.equal?(train) ? (train_count += 1) : (valid_count += 1)
end

train.close
valid.close

puts "Wrote #{train_count} train + #{valid_count} valid entries to #{OUT_DIR}/ (skipped #{skipped}, too long: #{skipped_long})"
