#!/usr/bin/env ruby
# frozen_string_literal: true

# Tests for tools/tokenizer.rb.
#
# Model-independent mechanics (byte map, pre-tokenizer pattern, chat
# template) always run. Golden-value tests (exact token ids/counts, verified
# against the original Python/transformers tokenizer) run only when a model
# directory is available: set TOKENIZER_MODEL_DIR, or run via bin/test
# (which auto-detects the Qwen3-8B MLX cache).

require_relative "helper"
require_relative "../tools/tokenizer"

MODEL_DIR = ENV["TOKENIZER_MODEL_DIR"].to_s.empty? ? nil : ENV["TOKENIZER_MODEL_DIR"]

# ---------------------------------------------------------------------------
# Mechanics — no model needed
# ---------------------------------------------------------------------------

# Byte map (GPT-2 bytes_to_unicode)
assert_equal "Ġ", Tokenizer::BYTE_MAP[32], "space byte maps to Ġ"
assert_equal "Ċ", Tokenizer::BYTE_MAP[10], "newline byte maps to Ċ"
assert_equal "Ĺ", Tokenizer::BYTE_MAP[0x97], "excluded byte 0x97 maps to Ĺ (256+offset)"
assert_equal "!", Tokenizer::BYTE_MAP[33], "printable ASCII maps to itself"
assert_equal "æ", Tokenizer::BYTE_MAP[0xE6], "Latin-1 byte maps to itself"
assert_equal 256, Tokenizer::BYTE_MAP.size, "byte map covers all 256 bytes"

# Pre-tokenizer pattern (GPT-2-style)
assert_equal ["Hello", " world"], "Hello world".scan(Tokenizer::PATTERN),
             "letters and space-prefixed words"
assert_equal ["1", "2", "3"], "123".scan(Tokenizer::PATTERN),
             "digits are split individually"
assert_equal ["don", "'t"], "don't".scan(Tokenizer::PATTERN),
             "contractions split off"
assert_equal ["\n\n", "Hi"], "\n\nHi".scan(Tokenizer::PATTERN),
             "newline runs and text"
assert_equal ["a", " b"], "a b".scan(Tokenizer::PATTERN),
             "single letters and leading-space words"

# Chat template (no tokenizer.json needed — allocate skips initialization)
tpl = Tokenizer.allocate
assert_equal(
  "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n",
  tpl.apply_chat_template([{ "role" => "system", "content" => "S" },
                           { "role" => "user", "content" => "U" }]),
  "system + user rendering"
)
assert_equal(
  "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello<|im_end|>\n",
  tpl.apply_chat_template([{ "role" => "user", "content" => "Hi" },
                           { "role" => "assistant", "content" => "Hello" }]),
  "final assistant wrapped in an empty <think> block (Qwen3 quirk)"
)
assert_equal(
  "<|im_start|>assistant\nx<|im_end|>\n",
  tpl.apply_chat_template([{ "role" => "assistant", "content" => "x" }]),
  "assistant-only chat gets no think block"
)
assert_equal(
  "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello<|im_end|>\n<|im_start|>assistant\n",
  tpl.apply_chat_template([{ "role" => "user", "content" => "Hi" },
                           { "role" => "assistant", "content" => "Hello" }],
                          add_generation_prompt: true),
  "generation prompt appended"
)
assert_equal(
  "<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\nfirst<|im_end|>\n" \
  "<|im_start|>user\nB<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nsecond<|im_end|>\n",
  tpl.apply_chat_template([{ "role" => "user", "content" => "A" },
                           { "role" => "assistant", "content" => "first" },
                           { "role" => "user", "content" => "B" },
                           { "role" => "assistant", "content" => "second" }]),
  "multi-turn: only the final assistant gets the think block"
)

# ---------------------------------------------------------------------------
# Golden values — need a model directory (verified against the original
# Python/transformers tokenizer for Qwen3-8B)
# ---------------------------------------------------------------------------

if MODEL_DIR.nil?
  puts "  (golden tests skipped — set TOKENIZER_MODEL_DIR or use bin/test)"
else
  tok = Tokenizer.new(MODEL_DIR)

  # Normal cases
  assert_equal [9707, 1879], tok.encode("Hello world"), "ids: 'Hello world'"
  assert_equal 2, tok.count("Hello world"), "count: 'Hello world'"
  assert_equal [101059, 102819, 56833, 61803, 70534], tok.encode("日本語テキスト"),
               "ids: CJK text"
  assert_equal 5, tok.count("日本語テキスト"), "count: CJK text"
  assert_equal [151644, 872, 198, 13048, 151645],
               tok.encode("<|im_start|>user\nHi<|im_end|>"),
               "ids: added tokens split atomically"
  assert_equal [], tok.encode(""), "empty text yields no tokens"

  # The Qwen3 vocab quirk: "!" is genuinely id 0 (first vocab entry)
  assert_equal [0], tok.encode("!"), "ids: '!' is the real id 0, not a fallback"

  # Full chat round-trip: templated ids (think block included)
  assert_equal(
    [151644, 872, 198, 9707, 1879, 151645, 198, 151644, 77091, 198,
     151667, 271, 151668, 271, 13048, 1052, 0, 2585, 646, 358, 1492, 30,
     151645, 198],
    tok.encode(tok.apply_chat_template(
      [{ "role" => "user", "content" => "Hello world" },
       { "role" => "assistant", "content" => "Hi there! How can I help?" }]
    )),
    "ids: full templated chat"
  )

  # Edge-case entries (counts verified against the Python tokenizer)
  edge_cases = [
    [["Hello world", "Hi there! How can I help?"], 24],
    [["", "Empty prompt test"], 17],
    [["🇺🇸🇪🇸 日本語テキスト 漢字とかな", "émoticônes et accents: café naïve"], 40],
    [["Show me code with <|im_start|> and <|im_end|> inside", "puts \"<|im_start|>literal\"  # special tokens in content"], 35],
    [["x = [1,2,3].map { |a| a * 2 }\nputs x.inspect\n# comment with ünïcödé", "def f(a,b)\n  a+b\nend\nf(1,2)"], 63],
    [["Line1\n\n\nLine4 with  lots  of  spaces", "\n\n\nTrailing and leading newlines\n\n"], 33],
    [["The quick brown fox jumps over the lazy dog 42 times.", "1234567890 !@#$%^&*()_+-=[]{};':\",./<>?`~"], 57],
    [["emoji: 😀🎉🚀🔥💯  flags: 🇺🇸🇩🇪🇯🇵  zwj: 👨‍👩‍👧‍👦  keycap: 1️⃣", "café résumé naïve élève — ünïcode"], 71],
    [["Tab\tcharacters\tand\tnon-breaking\u00a0spaces", "Mixed\u2028unicode\u2029separators and \u200bzero-width"], 36],
  ]
  edge_cases.each_with_index do |(pair, expected), i|
    msgs = [{ "role" => "user", "content" => pair[0] },
            { "role" => "assistant", "content" => pair[1] }]
    assert_equal expected, tok.count_messages(msgs), "edge case #{i + 1} count"
  end

  # Multi-turn + system message (MLX format)
  assert_equal 37, tok.count_messages(
    [{ "role" => "system", "content" => "You are a helpful assistant." },
     { "role" => "user", "content" => "MLX format with system message" },
     { "role" => "assistant", "content" => "MLX message format works too" }]
  ), "system + user + assistant count"
  assert_equal 33, tok.count_messages(
    [{ "role" => "user", "content" => "Two-turn chat" },
     { "role" => "assistant", "content" => "First answer" },
     { "role" => "user", "content" => "Follow-up" },
     { "role" => "assistant", "content" => "Second answer" }]
  ), "multi-turn count"
  assert_equal 9, tok.count_messages(
    [{ "role" => "assistant", "content" => "Chat starting with assistant" }]
  ), "assistant-only chat count"
end
