#!/usr/bin/env ruby
# frozen_string_literal: true

# A pragmatic Ruby re-implementation of a HuggingFace byte-level BPE
# tokenizer (the tokenizer.json format), reproducing the token counts of the
# original Python/transformers tokenizer closely for chat data.
#
# Implemented (the pieces that drive token counts):
#   - NFC normalization
#   - added/special tokens matched atomically (e.g. <|im_start|>)
#   - GPT-2-style pre-tokenization regex
#   - byte-level encoding (GPT-2 byte table)
#   - BPE merging from tokenizer.json merges, with a per-word cache
#   - chat-template application for the standard message structure
#     (system/user/assistant + generation prompt), including the Qwen3 quirk
#     of wrapping the final assistant message in an empty <think> block
#
# Not replicated (out of scope — affects only exotic inputs):
#   - tool-call / <tool_response> branches of the chat template
#   - reasoning_content handling beyond the empty-think quirk
#   - arbitrary third-party templates (tokenizer.json loading is model-
#     agnostic; the chat template here follows the Qwen convention)
#
# Usage as a library:
#   tok = Tokenizer.new(model_dir)
#   tok.count("some text")                     # token count of raw text
#   tok.encode("some text")                    # array of token ids
#   tok.apply_chat_template(msgs, add_generation_prompt: true)
#   tok.count_messages(msgs)                   # count of the templated chat
#
# Usage as a CLI:
#   printf 'text' | ruby tools/tokenizer.rb <model_dir> [--ids]
#   ruby tools/tokenizer.rb <model_dir> --chat '<messages_json>' [--ids]
# (--ids also prints the token ids; --chat tokenizes a chat instead of stdin)

require "json"

class Tokenizer
  # GPT-2-style pre-tokenization pattern (from the model's tokenizer.json).
  PATTERN = %r{(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+}
  # GPT-2 byte table: printable ASCII + Latin-1 map to themselves; all other
  # bytes map to U+0100+ in ascending byte order (space → "Ġ", newline → "Ċ",
  # byte 0x97 → "Ĺ"). Mirrors GPT-2's bytes_to_unicode() exactly.
  BYTE_MAP = begin
    include_set = ((33..126).to_a + (161..172).to_a + (174..255).to_a).freeze
    map = {}
    n = 0
    (0..255).each do |b|
      if include_set.include?(b)
        map[b] = b.chr(Encoding::UTF_8)
      else
        map[b] = (256 + n).chr(Encoding::UTF_8)
        n += 1
      end
    end
    map.freeze
  end

  attr_reader :vocab, :merges

  def initialize(model_dir)
    tj_path = File.join(model_dir, "tokenizer.json")
    abort "tokenizer.json not found in #{model_dir}" unless File.exist?(tj_path)

    tj = JSON.parse(File.read(tj_path))
    @vocab = tj["model"]["vocab"] # token → id
    @merges = tj["model"]["merges"] # ["A B", ...] or [["A", "B"], ...] in priority order
    @ranks = {}
    @merges.each_with_index do |m, i|
      key = m.is_a?(Array) ? m.join(" ") : m
      @ranks[key] = i
    end
    @added = {}
    tj["added_tokens"].each { |t| @added[t["content"]] = t["id"] }
    # Regexp.union treats string args as literals (escapes them itself), and
    # the capture group makes split() keep the matched separators.
    added_union = Regexp.union(@added.keys.sort_by { |k| -k.length })
    @added_split = Regexp.new("(#{added_union.source})")
    @cache = {}
  end

  # Full encode: normalize → added tokens → pre-tokenize → byte-level BPE.
  def encode(text)
    ids = []
    text.unicode_normalize(:nfc).split(@added_split, -1).each do |piece|
      if @added.key?(piece)
        ids << @added[piece]
        next
      end
      next if piece.empty?

      piece.scan(PATTERN).each do |word|
        bpe(byte_encode(word)).each { |t| ids << (@vocab[t] || 0) }
      end
    end
    ids
  end

  def count(text)
    encode(text).size
  end

  # Qwen-style chat template for the standard message structure. The final
  # assistant message is wrapped in an empty <think> block (Qwen3 template
  # behavior, when the conversation contains a user message).
  def apply_chat_template(messages, add_generation_prompt: false)
    has_user = messages.any? { |m| m["role"] == "user" }
    out = +""
    messages.each_with_index do |m, i|
      role = m["role"]
      content = m["content"].to_s
      case role
      when "system"
        out << "<|im_start|>system\n" << content << "<|im_end|>\n"
      when "user"
        out << "<|im_start|>user\n" << content << "<|im_end|>\n"
      when "assistant"
        out << "<|im_start|>assistant\n"
        if has_user && i == messages.size - 1
          out << "<think>\n\n</think>\n\n" << content.sub(/\A\n+/, "")
        else
          out << content
        end
        out << "<|im_end|>\n"
      end
    end
    out << "<|im_start|>assistant\n" if add_generation_prompt
    out
  end

  def count_messages(messages, add_generation_prompt: false)
    count(apply_chat_template(messages, add_generation_prompt: add_generation_prompt))
  end

  private

  def byte_encode(str)
    str.bytes.map { |b| BYTE_MAP[b] }.join
  end

  # Standard byte-level BPE with a per-word cache. Unknown merges stop the
  # loop; any piece missing from the vocab counts as one token (id 0).
  def bpe(token)
    return @cache[token] if @cache.key?(token)

    word = token.each_char.to_a
    return [token] if word.size == 1

    pairs = adjacent_pairs(word)
    loop do
      best = nil
      best_rank = Float::INFINITY
      pairs.each do |p|
        rank = @ranks[p]
        next unless rank && rank < best_rank

        best = p
        best_rank = rank
      end
      break if best.nil? || pairs.empty?

      word = merge_pair(word, best)
      pairs = adjacent_pairs(word)
    end

    @cache[token] = word
    word
  end

  def adjacent_pairs(word)
    word.each_cons(2).map { |a, b| "#{a} #{b}" }
  end

  def merge_pair(word, pair)
    a, b = pair.split(" ", 2)
    result = []
    i = 0
    while i < word.size
      if i < word.size - 1 && word[i] == a && word[i + 1] == b
        result << a + b
        i += 2
      else
        result << word[i]
        i += 1
      end
    end
    result
  end
end

if __FILE__ == $PROGRAM_NAME
  usage = "usage: #{$PROGRAM_NAME} <model_dir> [--ids] [--chat <messages_json>]"
  model_dir = ARGV.first
  abort usage if model_dir.nil? || model_dir.start_with?("--") || !File.directory?(model_dir)

  tok = Tokenizer.new(model_dir)
  show_ids = ARGV.include?("--ids")

  if (i = ARGV.index("--chat"))
    begin
      msgs = JSON.parse(ARGV[i + 1])
    rescue JSON::ParserError, TypeError
      abort usage
    end
    ids = tok.encode(tok.apply_chat_template(msgs))
    puts show_ids ? "#{ids.size} #{ids.inspect}" : ids.size
  else
    $stdin.each_line do |l|
      ids = tok.encode(l.chomp)
      puts show_ids ? "#{ids.size} #{ids.inspect}" : ids.size
    end
  end
end
