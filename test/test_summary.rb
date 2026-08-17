#!/usr/bin/env ruby
# frozen_string_literal: true

# Tests for tools/build_summary.rb — the distilled-knowledge generator.
# Runs against a sandbox fake repo (no model, no network).

require "rbconfig"
require "tempfile"
require "fileutils"
require "json"
require_relative "helper"

RUBY = RbConfig.ruby
SCRIPTS = File.expand_path("..", __dir__)

Dir.mktmpdir("summary_sandbox") do |sandbox|
  repo = File.join(sandbox, "fake_gem")
  FileUtils.mkdir_p(File.join(repo, "lib"))
  FileUtils.mkdir_p(File.join(repo, "test"))

  File.write(File.join(repo, "lib", "fake.rb"), <<~RUBY)
    # A fake gem used by the summary test.
    module Fake
      # Doubles the given number.
      def self.double(n)
        n * 2
      end
    end
  RUBY
  File.write(File.join(repo, "lib", "nodoc.rb"), "module NoDoc\n  def x; 1; end\nend\n")
  File.write(File.join(repo, "test", "fake_test.rb"), "class FakeTest\n  def test_x; end\nend\n")

  out = File.join(sandbox, "summary")
  ok = system(RUBY, File.join(SCRIPTS, "tools", "build_summary.rb"), repo, out)
  assert ok, "build_summary succeeds"

  jsonl = File.join(out, "fake_gem.jsonl")
  assert File.exist?(jsonl), "output file named after the repo"

  lines = File.readlines(jsonl)
  assert_equal 2, lines.size, "test files skipped; lib files with an extractable API kept"

  fake_entry = lines.find { |l| l.include?("lib/fake.rb") }
  assert fake_entry, "the documented lib file is distilled"
  convs = JSON.parse(fake_entry)["conversations"]
  assert_equal %w[human gpt], convs.map { |c| c["from"] }, "ShareGPT two-turn format"
  assert convs[0]["value"].include?("lib/fake.rb"), "prompt names the distilled file"
  assert convs[1]["value"].include?("Fake"), "answer carries the module"
  assert convs[1]["value"].include?("double"), "answer carries the public method"
end

# --- --max-entries cap ---
Dir.mktmpdir("summary_cap_sandbox") do |sandbox|
  repo = File.join(sandbox, "big_gem")
  FileUtils.mkdir_p(repo)
  3.times do |i|
    File.write(File.join(repo, "lib#{i}.rb"),
               "# Doc #{i}.\nmodule M#{i}\n  # Method #{i}.\n  def m#{i}; end\nend\n")
  end
  out = File.join(sandbox, "summary")
  ok = system(RUBY, File.join(SCRIPTS, "tools", "build_summary.rb"), repo, out, "--max-entries=2")
  assert ok, "build_summary with --max-entries succeeds"
  assert_equal 2, File.readlines(File.join(out, "big_gem.jsonl")).size, "entries capped at --max-entries"
end
