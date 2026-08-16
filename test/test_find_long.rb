#!/usr/bin/env ruby
# frozen_string_literal: true

# Tests for tools/find_long.rb. Estimator-mode tests always run; tokenizer-
# mode tests run when a model directory is available (set
# TOKENIZER_MODEL_DIR, or use bin/test which auto-detects the Qwen3-8B MLX
# cache).

require_relative "helper"
require_relative "../tools/find_long"

MODEL_DIR = ENV["TOKENIZER_MODEL_DIR"].to_s.empty? ? nil : ENV["TOKENIZER_MODEL_DIR"]
TOKENIZER = MODEL_DIR ? Tokenizer.new(MODEL_DIR) : nil

def entry(human, answer)
  JSON.generate("conversations" => [
    { "from" => "human", "value" => human },
    { "from" => "gpt", "value" => answer },
  ])
end

BIG = entry("Show me big.rb", "y" * 20_000) # estimator: ~14.3k est tokens
NORMAL = entry("short prompt", "z" * 500)   # estimator: ~430 est tokens
MSGS = JSON.generate("messages" => [        # estimator: ~2.2k est tokens (over)
  { "role" => "user", "content" => "hi" },
  { "role" => "assistant", "content" => "w" * 3000 },
])
BAD_JSON = "{not valid json"

LINES = [BIG, NORMAL, MSGS, "", BAD_JSON].freeze

# --- estimator mode ---
results = quietly { find_over_limit(LINES, max_len: MAX_CONTEXT_TOKENS, limit: 20, tokenizer: nil) }
assert_equal [1, 3], results.map { |r| r[:line] },
             "flags the two over-limit entries (lines 1 and 3), skips empty/bad"
assert results.all? { |r| r[:len] > MAX_CONTEXT_TOKENS }, "all flagged lengths exceed the cap"
assert_equal 20_014, results[0][:chars], "chars = total content bytes"
assert_equal "Show me big.rb", results[0][:prompt], "prompt preview is the first message"

limited = quietly { find_over_limit(LINES, max_len: MAX_CONTEXT_TOKENS, limit: 1, tokenizer: nil) }
assert_equal [1], limited.map { |r| r[:line] }, "limit caps the number of results"

none = find_over_limit([NORMAL], max_len: MAX_CONTEXT_TOKENS, limit: 20, tokenizer: nil)
assert_equal [], none, "nothing flagged when everything fits"

# --- tokenizer mode (golden: real counts verified against the reference) ---
if TOKENIZER
  res = find_over_limit([BIG, NORMAL, MSGS], max_len: MAX_CONTEXT_TOKENS, limit: 20,
                        tokenizer: TOKENIZER)
  assert_equal [1], res.map { |r| r[:line] },
               "tokenizer mode: only the 20k-char entry is really over 2048"
  assert res[0][:len] > MAX_CONTEXT_TOKENS, "real token count exceeds the cap"
else
  puts "  (tokenizer-mode tests skipped — set TOKENIZER_MODEL_DIR or use bin/test)"
end
