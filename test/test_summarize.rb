#!/usr/bin/env ruby
# frozen_string_literal: true

# Tests for bin/summarize — capsules every repo under _sources/. Runs in a
# sandbox (no model, no network).

require "rbconfig"
require "tempfile"
require "fileutils"
require "json"
require_relative "helper"

RUBY = RbConfig.ruby
SCRIPTS = File.expand_path("..", __dir__)

def fake_repo(dir, name, code)
  repo = File.join(dir, "_sources", name)
  FileUtils.mkdir_p(File.join(repo, "lib"))
  File.write(File.join(repo, "lib", "#{name}.rb"), code)
  repo
end

# --- two repos → one capsule each ---
Dir.mktmpdir("summarize_two") do |sandbox|
  fake_repo(sandbox, "alpha", "# Alpha.\nmodule Alpha\n  # Doubles.\n  def self.double(n)\n    n * 2\n  end\nend\n")
  fake_repo(sandbox, "beta", "# Beta.\nmodule Beta\n  # Triples.\n  def self.triple(n)\n    n * 3\n  end\nend\n")

  env = { "LLM_TRAINER_ROOT" => sandbox }
  ok = system(env, RUBY, File.join(SCRIPTS, "bin", "summarize"),
              out: File::NULL, err: File::NULL)
  assert ok, "bin/summarize succeeds"

  summary_dir = File.join(sandbox, "summary")
  assert File.exist?(File.join(summary_dir, "alpha.jsonl")), "alpha capsule written"
  assert File.exist?(File.join(summary_dir, "beta.jsonl")), "beta capsule written"

  alpha = File.readlines(File.join(summary_dir, "alpha.jsonl"))
  assert_equal 1, alpha.size, "one distilled entry per repo"
  convs = JSON.parse(alpha.first)["conversations"]
  assert convs[1]["value"].include?("Alpha"), "alpha capsule carries its API"
end

# --- one undistillable repo: warn, continue, non-zero exit ---
Dir.mktmpdir("summarize_partial") do |sandbox|
  good = fake_repo(sandbox, "good", "# Good.\nmodule Good\n  # Pings.\n  def self.ping\n    :pong\n  end\nend\n")
  bad = File.join(sandbox, "_sources", "bad")
  FileUtils.mkdir_p(bad)
  File.write(File.join(bad, "plain.txt"), "no ruby here")

  env = { "LLM_TRAINER_ROOT" => sandbox }
  out = IO.popen([env, RUBY, File.join(SCRIPTS, "bin", "summarize")], err: [:child, :out], &:read)
  status = $?
  assert !status.success?, "non-zero exit when a repo cannot be summarized"
  assert out.include?("FAILED to summarize: bad"), "failing repo named"
  assert File.exist?(File.join(sandbox, "summary", "good.jsonl")), "the good repo still got its capsule"
end

# --- empty _sources: friendly failure ---
Dir.mktmpdir("summarize_empty") do |sandbox|
  FileUtils.mkdir_p(File.join(sandbox, "_sources"))
  env = { "LLM_TRAINER_ROOT" => sandbox }
  out = IO.popen([env, RUBY, File.join(SCRIPTS, "bin", "summarize")], err: [:child, :out], &:read)
  status = $?
  assert !status.success?, "non-zero exit with no repos"
  assert out.include?("No repositories found"), "error names the missing repos"
end
