#!/usr/bin/env ruby
# frozen_string_literal: true

# Orchestrates the whole training-data pipeline in one go:
#
#   1. build_docs_context.rb — git pull every repo under _sources/, generate docs_context/
#   2. build_code_context.rb — git pull every repo in _sources/, generate
#      code_context/<repo>.md + _dataset/code_<repo>.jsonl
#   3. create_dataset.rb     — generate docs datasets + _full_ruby_dataset.jsonl
#   4. build_attribution.rb   — regenerate Attribution.md (git-ignored)
#
# After it finishes, _dataset/ is populated and _full_ruby_dataset.jsonl can
# be fed to a local model-training pipeline as-is (ShareGPT-format JSONL).
#
# Any step that fails aborts the run, since the later steps depend on the
# earlier ones' output.
#
# Usage:
#   ruby main.rb

require "rbconfig"

ROOT = File.dirname(__FILE__)
RUBY = RbConfig.ruby

STEPS = [
  ["Build docs context", "build_docs_context.rb"],
  ["Build code context", "build_code_context.rb"],
  ["Build datasets",     "create_dataset.rb"],
  ["Build attribution",  "build_attribution.rb"],
].freeze

def run_step(label, script)
  puts
  puts "=== #{label} (#{script}) ==="
  ok = system(RUBY, File.join(ROOT, script))
  abort "FAILED: #{script}" unless ok
end

started_at = Time.now
puts "Pipeline started at #{started_at.strftime("%Y-%m-%d %H:%M:%S")}"

STEPS.each { |label, script| run_step(label, script) }

elapsed = Time.now - started_at
puts
puts "Done! _dataset/ populated (#{elapsed.round(1)}s)."
puts "Train a local model with: _dataset/_full_ruby_dataset.jsonl"
