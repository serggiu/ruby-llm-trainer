#!/usr/bin/env ruby
# frozen_string_literal: true

# Tests for bin/refresh — the knowledge-capsule refresh task. Runs against
# sandboxes (no model, no MLX needed).

require "rbconfig"
require "tempfile"
require "fileutils"
require_relative "helper"

RUBY = RbConfig.ruby
SCRIPTS = File.expand_path("..", __dir__)

# --- plan building (loaded as a library) ---
load File.join(SCRIPTS, "bin", "refresh")

def plan_for(*args, data_root: SCRIPTS)
  build_refresh_plan(args, scripts: SCRIPTS, data_root: data_root,
                     model_cache_glob: "/nonexistent/*/tokenizer.json")
end

# --- defaults ---
Dir.mktmpdir("refresh_plan") do |sandbox|
  summary = File.join(sandbox, "summary")
  FileUtils.mkdir_p(summary)
  File.write(File.join(summary, "minitest.jsonl"), "{}\n")
  File.write(File.join(summary, "timecop.jsonl"), "{}\n")

  p = plan_for("--model", "/models/fake", data_root: sandbox)
  assert_equal 200, p[:iters], "default refresh iters"
  assert_equal 2, p[:capsules].size, "all capsules picked up"
  assert p[:slice_dir].end_with?("_mlx/qwen3_refresh"), "refresh slice dir"
  assert p[:combined].end_with?("_mlx/summary_combined.jsonl"), "combined capsule file"
  assert p[:log_path].end_with?("_mlx/train_refresh.log"), "refresh log path"
  assert p[:run_cmd].include?("--resume-adapter-file"), "refresh always resumes"
  assert p[:run_cmd].include?(p[:lora]), "refresh uses the venv mlx_lm.lora"
  assert_equal false, p[:watchdog], "watchdog off by default"

  p2 = plan_for("--model", "/models/fake", "--iters", "50", "--watchdog", "--dry-run", data_root: sandbox)
  assert_equal 50, p2[:iters], "--iters parsed"
  assert_equal true, p2[:watchdog], "--watchdog parsed"
  assert_equal true, p2[:dry_run], "--dry-run parsed"
end

# --- no capsules: friendly failure ---
Dir.mktmpdir("no_capsules") do |sandbox|
  env = { "LLM_TRAINER_ROOT" => sandbox, "TOKENIZER_MODEL_DIR" => "" }
  out = IO.popen([env, RUBY, File.join(SCRIPTS, "bin", "refresh"),
                  "--dry-run", "--model", "/models/fake"], err: [:child, :out], &:read)
  status = $?
  assert !status.success?, "bin/refresh exits non-zero without capsules"
  assert out.include?("no knowledge capsules found"), "error names the missing capsules"
  assert out.include?("build_summary.rb"), "error points to the summary tool"
  assert !out.include?("Errno"), "no raw exception leaks to the user"
end

# --- capsules present but no trained model: friendly failure ---
Dir.mktmpdir("no_model") do |sandbox|
  FileUtils.mkdir_p(File.join(sandbox, "summary"))
  File.write(File.join(sandbox, "summary", "minitest.jsonl"), "{}\n")
  env = { "LLM_TRAINER_ROOT" => sandbox, "TOKENIZER_MODEL_DIR" => "" }
  out = IO.popen([env, RUBY, File.join(SCRIPTS, "bin", "refresh"),
                  "--dry-run", "--model", "/models/fake"], err: [:child, :out], &:read)
  status = $?
  assert !status.success?, "bin/refresh exits non-zero without a trained model"
  assert out.include?("No trained model found"), "error names the missing adapter"
  assert out.include?("ruby bin/train"), "error tells the user to train first"
end

# --- happy path: capsules + trained model, dry-run succeeds ---
Dir.mktmpdir("refresh_ready") do |sandbox|
  FileUtils.mkdir_p(File.join(sandbox, "summary"))
  FileUtils.mkdir_p(File.join(sandbox, "_mlx", "adapters_qwen3"))
  File.write(File.join(sandbox, "summary", "minitest.jsonl"), "{}\n")
  File.write(File.join(sandbox, "_mlx", "adapters_qwen3", "adapters.safetensors"), "weights")
  env = { "LLM_TRAINER_ROOT" => sandbox, "TOKENIZER_MODEL_DIR" => "" }
  out = IO.popen([env, RUBY, File.join(SCRIPTS, "bin", "refresh"),
                  "--dry-run", "--model", "/models/fake"], err: [:child, :out], &:read)
  status = $?
  assert status.success?, "bin/refresh --dry-run succeeds with capsules + trained model"
  assert out.include?("minitest.jsonl"), "dry-run lists the capsules"
  assert out.include?("--resume-adapter-file"), "dry-run shows the resume flag"
end

# --- combined capsule write ---
Dir.mktmpdir("combined") do |sandbox|
  a = File.join(sandbox, "a.jsonl")
  b = File.join(sandbox, "b.jsonl")
  File.write(a, "line-a1\nline-a2\n")
  File.write(b, "line-b1\n")
  combined = File.join(sandbox, "out", "combined.jsonl")
  write_combined_capsules([a, b], combined)
  assert_equal "line-a1\nline-a2\nline-b1\n", File.read(combined), "capsules concatenated in order"
end
