#!/usr/bin/env ruby
# frozen_string_literal: true

# Tests for the context-length bounding logic in build_dataset.rb
# (MAX_CONTEXT_TOKENS, split_text_chunks, split_long_entries) and for the
# test-coverage extraction (build_coverage, Rails + minitest styles).

require_relative "helper"
require_relative "../build/build_dataset"

def build_long_answer
  (1..900).map { |i| "def method_#{i}\n  # doc for #{i}\n  #{i} * 2\nend\n" }.join
end

def build_long_prompt
  (1..600).map { |i| "test_#{i} \"x\" do\n  assert true\nend\n" }.join
end

def max_pair_tokens(entries)
  entries.map { |h, a| estimate_tokens(h) + estimate_tokens(a) }.max
end

cap = MAX_CONTEXT_TOKENS - CHAT_TEMPLATE_OVERHEAD_TOKENS

# --- 1. Long answer, short prompt (the common case) ---
long_answer = build_long_answer
entries = split_long_entries("Show me the complete Ruby source code of `big.rb`.", long_answer)
assert entries.size > 1, "long answer is split into multiple pairs"
assert max_pair_tokens(entries) <= cap, "every pair fits under the cap"
assert entries.all? { |h, _| h.include?("(part ") }, "prompts annotated with part numbers"
assert_equal long_answer, entries.map { |_, a| a }.join, "no content lost or duplicated"

# --- 2. Long prompt, short answer (e.g. huge test file as prompt) ---
long_prompt = build_long_prompt
short_answer = "class Thing\nend\n"
entries2 = split_long_entries(long_prompt, short_answer)
assert entries2.size > 1, "long prompt is split into multiple pairs"
assert max_pair_tokens(entries2) <= cap, "every pair fits under the cap"
assert entries2.all? { |_, a| a == short_answer }, "answer repeated per prompt part"

# --- 3. Both long: split both, paired part-by-part ---
entries3 = split_long_entries(build_long_prompt, build_long_answer)
assert entries3.size > 1, "both-long case produces multiple pairs"
assert max_pair_tokens(entries3) <= cap, "every pair fits under the cap"

# --- 4. Short pair stays untouched ---
assert_equal [["short prompt", "short answer"]],
             split_long_entries("short prompt", "short answer"),
             "short pair passes through unchanged"

# --- 5. Single over-long line (hard split, must reassemble exactly) ---
giant = "x" * 100_000
entries4 = split_long_entries("p", giant)
assert entries4.size > 1, "giant line is hard-split"
assert max_pair_tokens(entries4) <= cap, "every pair fits under the cap"
assert_equal giant, entries4.map { |_, a| a }.join, "giant line reassembles exactly"

# --- 6. Empty sides are returned as-is (never dropped) ---
assert_equal [["", "answer"]], split_long_entries("", "answer"), "empty prompt kept"
assert_equal [["prompt", ""]], split_long_entries("prompt", ""), "empty answer kept"

# --- 7. Coverage extraction: Rails `test "name"` declarations ---
rails_code = <<~RUBY
  class WidgetTest < ActiveSupport::TestCase
    test "creates a widget" do
      assert Widget.create
    end

    test "destroys a widget" do
      assert true
    end
  end
RUBY
rails_coverage = build_coverage("test/widget_test.rb", rails_code)
assert rails_coverage.include?("creates a widget"), "Rails test name captured"
assert rails_coverage.include?("destroys a widget"), "second Rails test name captured"

# --- 8. Coverage extraction: minitest `def test_name` methods ---
minitest_code = <<~RUBY
  class TestWidget < Minitest::Test
    def setup
      @widget = Widget.new
    end

    def test_creates_widget
      assert @widget.valid?
    end

    def test_destroys_widget?
      assert true
    end

    def helper_not_a_test
      true
    end
  end
RUBY
minitest_coverage = build_coverage("test/test_widget.rb", minitest_code)
assert minitest_coverage.include?("test_creates_widget"), "minitest method captured"
assert minitest_coverage.include?("test_destroys_widget?"), "minitest method with ? captured"
assert !minitest_coverage.include?("setup"), "non-test methods not listed"
assert !minitest_coverage.include?("helper_not_a_test"), "helper methods not listed"

# --- 9. Coverage extraction: no tests → nil (file skipped) ---
plain = "class Foo\n  def bar\n    1\n  end\nend\n"
assert_nil build_coverage("lib/foo.rb", plain), "no tests means no coverage entry"

# --- 9b. recommended_iters mirrors bin/train's auto-split ---
assert_equal 2183, recommended_iters(2729), "one epoch = train portion after ~20% holdout"
assert_equal 1264, recommended_iters(1580), "railties-size dataset"
assert_equal 164, recommended_iters(205), "small dataset"
assert_equal [6, 6], [recommended_iters(12), 12 - recommended_iters(12)], "tiny set split"
assert_equal 1, recommended_iters(1), "single entry"

# --- 10. test_file? path detection (test/ and spec/) ---
assert test_file?("test/models/room_test.rb"), "test/ paths are test files"
assert test_file?("spec/models/room_spec.rb"), "spec/ paths are test files"
assert !test_file?("lib/foo.rb"), "lib/ paths are not test files"
assert !test_file?("app/foo.rb"), "app/ paths are not test files"

# --- 11. Coverage extraction: RSpec `it`/`specify`/`example` blocks ---
rspec_code = <<~RUBY
  RSpec.describe Demo do
    describe "#double" do
      context "with a number" do
        it "doubles the input" do
          expect(Demo.double(2)).to eq(4)
        end

        it "doesn't raise on nil" do
          expect(Demo.double(nil)).to be_nil
        end

        specify "handles zero" do
          expect(Demo.double(0)).to eq(0)
        end

        it('handles parens')

        it { is_expected.to be_truthy }
      end
    end
  end
RUBY
rspec_coverage = build_coverage("spec/demo_spec.rb", rspec_code)
assert rspec_coverage.include?("doubles the input"), "RSpec it block captured"
assert rspec_coverage.include?("doesn't raise on nil"), "apostrophe inside double quotes kept"
assert rspec_coverage.include?("handles zero"), "specify alias captured"
assert rspec_coverage.include?("handles parens"), "parenthesized it captured"
assert !rspec_coverage.include?("#double"), "describe/context headings not listed"
assert !rspec_coverage.include?("is_expected"), "block-form it without a name not listed"
assert !rspec_coverage.include?("Demo"), "RSpec.describe not listed"

puts "  (#{entries.size} + #{entries2.size} + #{entries3.size} + #{entries4.size} pairs generated across split cases)"
