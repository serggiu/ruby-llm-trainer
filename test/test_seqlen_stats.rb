#!/usr/bin/env ruby
# frozen_string_literal: true

# Tests for tools/seqlen_stats.rb. Estimator-mode tests always run;
# tokenizer-mode tests run when a model directory is available (set
# TOKENIZER_MODEL_DIR, or use bin/test which auto-detects the Qwen3-8B MLX
# cache).

require_relative "helper"
require_relative "../tools/seqlen_stats"

MODEL_DIR = ENV["TOKENIZER_MODEL_DIR"].to_s.empty? ? nil : ENV["TOKENIZER_MODEL_DIR"]
TOKENIZER = MODEL_DIR ? Tokenizer.new(MODEL_DIR) : nil

def entry(human, answer)
  JSON.generate("conversations" => [
    { "from" => "human", "value" => human },
    { "from" => "gpt", "value" => answer },
  ])
end

LONG = entry("a", "b" * 4000) # est = ceil(4002/1.4) + 64 = 2923
MID = entry("c", "d" * 1000)  # est = 716 + 64 = 780
SHORT = entry("e", "f" * 200) # est = 145 + 64 = 209
BAD = "{bad"

# --- estimator mode ---
lens, bad = quietly { collect_lengths([LONG, MID, SHORT, BAD, ""], tokenizer: nil) }
assert_equal 3, lens.size, "bad and empty lines excluded from lengths"
assert_equal 1, bad, "one malformed line counted as skipped"

lens.sort!
s = summarize(lens, MAX_CONTEXT_TOKENS, real: false)
assert_equal 3, s[:samples], "sample count"
assert_equal [209, 780, 2923], [s[:min], s[:p50], s[:max]], "min/p50/max on the sorted set"
assert_equal 1304, s[:mean], "integer mean of 209 + 780 + 2923"
assert_equal 2923, s[:p90], "p90 clamps to max on small sets"
assert_equal 1, s[:over], "one entry over the cap"
assert_equal 33.3, s[:over_pct].round(1), "over percentage"
assert_equal "est tokens", s[:label], "estimator label"

empty_lens, empty_bad = collect_lengths([], tokenizer: nil)
assert_equal 0, empty_lens.size, "empty input yields no lengths"
assert_equal 0, empty_bad, "empty input yields no skipped"

# --- tokenizer mode (golden: counts verified against the reference) ---
if TOKENIZER
  golden = [
    entry("Hello world", "Hi there! How can I help?"),                        # 24 tokens
    entry("", "Empty prompt test"),                                           # 17 tokens
    entry("🇺🇸🇪🇸 日本語テキスト 漢字とかな", "émoticônes et accents: café naïve"), # 40 tokens
  ]
  glens, gbad = collect_lengths(golden, tokenizer: TOKENIZER)
  glens.sort!
  g = summarize(glens, MAX_CONTEXT_TOKENS, real: true)
  assert_equal 0, gbad, "golden entries all parse"
  assert_equal [17, 24, 40], [g[:min], g[:p50], g[:max]],
               "golden min/p50/max match the reference tokenizer"
  assert_equal 27, g[:mean], "golden mean of 17 + 24 + 40"
  assert_equal 0, g[:over], "golden entries all under the cap"
  assert_equal "tokens", g[:label], "real-tokenizer label"
else
  puts "  (tokenizer-mode tests skipped — set TOKENIZER_MODEL_DIR or use bin/test)"
end
