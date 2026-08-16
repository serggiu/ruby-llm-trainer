#!/usr/bin/env ruby
# frozen_string_literal: true

# Tests for bin/train's plan building (no model, no MLX needed) and for
# build_mlx_data.rb running in a sandbox (LLM_TRAINER_ROOT).

require "rbconfig"
require "tempfile"
require "fileutils"
require "json"
require_relative "helper"

# Loads bin/train as a library: defines SCRIPTS/ROOT/SLICES/build_plan;
# the __FILE__ guard keeps its main flow from running.
load File.join(File.expand_path("..", __dir__), "bin", "train")

def plan_for(*args, data_root: SCRIPTS)
  build_plan(args, scripts: SCRIPTS, data_root: data_root,
                   model_cache_glob: "/nonexistent/*/tokenizer.json")
end

MODEL = "/models/fake"

# --- defaults ---
p = plan_for("--model", MODEL)
assert_equal 1000, p[:iters], "default iters"
assert_equal "proper", p[:slice], "default slice"
assert_equal [3000, 200], [p[:train_count], p[:valid_count]], "proper slice sizes"
assert p[:slice_dir].end_with?("_mlx/qwen3_proper"), "slice dir under the data root"
assert p[:sft_set].end_with?("_dataset/sft_train_set.jsonl"), "sft set under the data root"
assert p[:log_path].end_with?("_mlx/train_proper.log"), "log path under the data root, named by slice"
assert_equal false, p[:watchdog], "watchdog off by default"
assert_equal false, p[:dry_run], "dry-run off by default"
assert p[:build_cmd].any? { |x| x.include?("build_mlx_data.rb") }, "build command targets build_mlx_data"
assert p[:build_cmd].include?("3000") && p[:build_cmd].include?("200"), "build command carries the counts"
assert p[:run_cmd].any? { |x| x.include?("run_capped.rb") }, "training runs through the watchdog wrapper"
assert p[:run_cmd].include?(p[:lora]), "training command uses the venv mlx_lm.lora"
assert p[:run_cmd].include?("--data") && p[:run_cmd].include?(p[:slice_dir]), "training targets the slice"
assert p[:run_cmd].include?("--max-seq-length") && p[:run_cmd].include?("2048"), "seq length capped at 2048"
assert p[:run_cmd].include?("--val-batches") && p[:run_cmd].include?("10"), "eval tweak applied"

# --- flags ---
p2 = plan_for("--model", MODEL, "--iters", "500", "--slice", "full",
              "--adapter", "_mlx/my_adapter", "--watchdog", "--dry-run")
assert_equal 500, p2[:iters], "--iters parsed"
assert_equal "full", p2[:slice], "--slice parsed"
assert_equal [20_000, 700], [p2[:train_count], p2[:valid_count]], "full slice sizes"
assert_equal "_mlx/my_adapter", p2[:adapter], "--adapter parsed"
assert_equal true, p2[:watchdog], "--watchdog parsed"
assert_equal true, p2[:dry_run], "--dry-run parsed"
assert p2[:run_cmd].include?("--iters") && p2[:run_cmd].include?("500"), "iters in the command"

# --- errors (clear TOKENIZER_MODEL_DIR so the fallbacks are exercised) ---
old_env = ENV["TOKENIZER_MODEL_DIR"]
ENV["TOKENIZER_MODEL_DIR"] = ""
assert_raises(ArgumentError, "unknown option rejected") { plan_for("--bogus") }
assert_raises(ArgumentError, "unknown slice rejected") { plan_for("--slice", "nope") }
assert_raises(ArgumentError, "missing --model value rejected") { plan_for("--model") }

# --- model auto-detection fallback ---
Dir.mktmpdir("fake_model") do |fake_model|
  File.write(File.join(fake_model, "tokenizer.json"), "{}")
  p3 = build_plan(["--slice", "smoke"], scripts: SCRIPTS, data_root: SCRIPTS,
                  model_cache_glob: File.join(fake_model, "tokenizer.json"))
  assert_equal fake_model, p3[:model], "model auto-detected from the cache glob"
end
ENV["TOKENIZER_MODEL_DIR"] = old_env

# --- build_mlx_data in a sandbox (no model needed) ---
Dir.mktmpdir("mlx_sandbox") do |sandbox|
  dataset = File.join(sandbox, "_dataset")
  FileUtils.mkdir_p(dataset)
  entry = lambda do |h, a|
    JSON.generate("conversations" => [
      { "from" => "human", "value" => h },
      { "from" => "gpt", "value" => a },
    ])
  end
  File.write(File.join(dataset, "sft_train_set.jsonl"),
             [entry.call("one", "a" * 100), entry.call("two", "b" * 100),
              entry.call("three", "c" * 100)].join("\n") + "\n")

  ok = system({ "LLM_TRAINER_ROOT" => sandbox }, RbConfig.ruby,
              File.join(SCRIPTS, "build", "build_mlx_data.rb"), "2", "1",
              out: File::NULL, err: File::NULL)
  assert ok, "build_mlx_data runs in the sandbox with default paths"

  train = File.join(sandbox, "_mlx", "train.jsonl")
  valid = File.join(sandbox, "_mlx", "valid.jsonl")
  assert File.exist?(train) && File.exist?(valid), "train/valid written under the data root"
  assert_equal 2, File.readlines(train).size, "train slice takes the first 2 entries"
  assert_equal 1, File.readlines(valid).size, "valid slice takes the next entry"
  first_msg = JSON.parse(File.readlines(train).first)["messages"]
  assert_equal %w[user assistant], first_msg.map { |m| m["role"] }, "MLX messages format"
end

# --- run_with_log tees output to a file ---
Dir.mktmpdir("log_sandbox") do |d|
  log = File.join(d, "train.log")
  ok = run_with_log(["printf", "line one\nline two\n"], log)
  assert ok, "run_with_log succeeds"
  assert_equal "line one\nline two\n", File.read(log), "run_with_log tees all output to the log"
end

# --- missing training data: friendly failure, not a crash ---
Dir.mktmpdir("empty_sandbox") do |sandbox|
  env = { "LLM_TRAINER_ROOT" => sandbox, "TOKENIZER_MODEL_DIR" => "" }
  out = IO.popen([env, RbConfig.ruby, File.join(SCRIPTS, "bin", "train"),
                  "--dry-run", "--model", MODEL], err: [:child, :out], &:read)
  status = $?
  assert !status.success?, "bin/train exits non-zero when the training data is missing"
  assert out.include?("Training data not found"), "error names the missing data"
  assert out.include?("sft_train_set.jsonl"), "error names the missing file"
  assert out.include?("ruby bin/build"), "error tells the user to run bin/build first"
  assert !out.include?("Errno"), "no raw exception leaks to the user"
end

# --- with training data present, dry-run succeeds ---
Dir.mktmpdir("ready_sandbox") do |sandbox|
  dataset = File.join(sandbox, "_dataset")
  FileUtils.mkdir_p(dataset)
  File.write(File.join(dataset, "sft_train_set.jsonl"),
             JSON.generate("conversations" => [
               { "from" => "human", "value" => "h" },
               { "from" => "gpt", "value" => "a" },
             ]) + "\n")
  env = { "LLM_TRAINER_ROOT" => sandbox, "TOKENIZER_MODEL_DIR" => "" }
  ok = system(env, RbConfig.ruby, File.join(SCRIPTS, "bin", "train"),
              "--dry-run", "--model", MODEL, out: File::NULL, err: File::NULL)
  assert ok, "bin/train --dry-run succeeds when the training data exists"
end
