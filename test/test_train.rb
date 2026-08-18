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
assert_equal false, p[:resume], "resume off by default"
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
              "--adapter", "_mlx/my_adapter", "--watchdog", "--resume", "--dry-run")
assert_equal 500, p2[:iters], "--iters parsed"
assert_equal "full", p2[:slice], "--slice parsed"
assert_equal [20_000, 700], [p2[:train_count], p2[:valid_count]], "full slice sizes"
assert_equal "_mlx/my_adapter", p2[:adapter], "--adapter parsed"
assert_equal true, p2[:watchdog], "--watchdog parsed"
assert_equal true, p2[:resume], "--resume parsed"
assert_equal true, p2[:dry_run], "--dry-run parsed"
assert p2[:run_cmd].include?("--iters") && p2[:run_cmd].include?("500"), "iters in the command"
assert p2[:run_cmd].include?("--resume-adapter-file"), "resume flag in the command"
assert p2[:resume_file].end_with?("_mlx/my_adapter/adapters.safetensors"), "resume targets the adapter checkpoint"

# --- --restore parsing ---
p3 = plan_for("--model", MODEL, "--restore", "_mlx/snapshots/pre_proper_20260817_120000")
assert_equal "_mlx/snapshots/pre_proper_20260817_120000", p3[:restore], "--restore parsed"

# --- --force parsing and conflicts ---
p4 = plan_for("--model", MODEL, "--force")
assert_equal true, p4[:force], "--force parsed"
assert_raises(ArgumentError, "--resume and --force conflict rejected") { plan_for("--model", MODEL, "--resume", "--force") }

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
  assert_equal 2, File.readlines(train).size, "train slice takes 2 entries"
  assert_equal 1, File.readlines(valid).size, "valid slice takes 1 entry"
  first_msg = JSON.parse(File.readlines(train).first)["messages"]
  assert_equal %w[user assistant], first_msg.map { |m| m["role"] }, "MLX messages format"

  # --- seeded shuffle: reproducible split, content preserved ---
  pairs_of = lambda do |dir|
    (Dir.glob(File.join(dir, "_mlx", "{train,valid}.jsonl")).flat_map do |f|
      File.readlines(f).map do |l|
        m = JSON.parse(l)["messages"]
        [m[0]["content"], m[1]["content"]]
      end
    end).sort
  end
  expected_pairs = [["one", "a" * 100], ["two", "b" * 100], ["three", "c" * 100]].sort
  assert_equal expected_pairs, pairs_of.call(sandbox), "every input entry lands in train or valid"

  run_build = lambda do |dir, seed_flag|
    FileUtils.rm_rf(File.join(dir, "_mlx"))
    system({ "LLM_TRAINER_ROOT" => dir }, RbConfig.ruby,
           File.join(SCRIPTS, "build", "build_mlx_data.rb"), "2", "1", seed_flag,
           out: File::NULL, err: File::NULL) or raise "build failed"
    File.read(File.join(dir, "_mlx", "train.jsonl")) + File.read(File.join(dir, "_mlx", "valid.jsonl"))
  end
  first_run = run_build.call(sandbox, "--shuffle-seed=7")
  assert_equal first_run, run_build.call(sandbox, "--shuffle-seed=7"),
               "same seed → same split (reproducible)"
  assert first_run.start_with?(%({"messages":)), "shuffled output still valid MLX JSON"
  first_train = JSON.parse(File.readlines(File.join(sandbox, "_mlx", "train.jsonl")).first)["messages"]
  assert_equal "c" * 100, first_train[1]["content"],
               "seed 7 changes the split order (the third input entry lands first)"
end

# --- snapshot_adapter: copies the current model state ---
Dir.mktmpdir("snap_sandbox") do |sandbox|
  adapter_dir = File.join(sandbox, "_mlx", "adapters_qwen3")
  FileUtils.mkdir_p(adapter_dir)
  File.write(File.join(adapter_dir, "adapters.safetensors"), "weights-v1")
  File.write(File.join(adapter_dir, "0000100_adapters.safetensors"), "ckpt-v1")

  snap = snapshot_adapter(adapter_dir, "proper", File.join(sandbox, "_mlx", "snapshots"))
  assert snap, "snapshot created when an adapter exists"
  assert snap.include?("snapshots/pre_proper_"), "snapshot named pre_<slice>_<timestamp>"
  assert_equal "weights-v1", File.read(File.join(snap, "adapters.safetensors")), "snapshot carries the weights"
  assert_equal "ckpt-v1", File.read(File.join(snap, "0000100_adapters.safetensors")), "snapshot carries the checkpoints"

  empty = File.join(sandbox, "_mlx", "no_adapter_yet")
  assert_nil snapshot_adapter(empty, "proper", File.join(sandbox, "_mlx", "snapshots")),
             "no snapshot on a first-ever run"
end

# --- --restore: rolls back to a saved state (subprocess) ---
Dir.mktmpdir("restore_sandbox") do |sandbox|
  adapter_dir = File.join(sandbox, "_mlx", "adapters_qwen3")
  snap = File.join(sandbox, "_mlx", "snapshots", "pre_proper_20260817_120000")
  FileUtils.mkdir_p(adapter_dir)
  FileUtils.mkdir_p(snap)
  File.write(File.join(adapter_dir, "adapters.safetensors"), "bad-session-weights")
  File.write(File.join(snap, "adapters.safetensors"), "good-saved-weights")

  env = { "LLM_TRAINER_ROOT" => sandbox, "TOKENIZER_MODEL_DIR" => "" }
  ok = system(env, RbConfig.ruby, File.join(SCRIPTS, "bin", "train"),
              "--restore", snap, "--model", MODEL, out: File::NULL, err: File::NULL)
  assert ok, "bin/train --restore succeeds"
  assert_equal "good-saved-weights", File.read(File.join(adapter_dir, "adapters.safetensors")),
               "adapter dir rolled back to the snapshot"
end

# --- --restore with a missing snapshot: friendly failure ---
Dir.mktmpdir("bad_restore_sandbox") do |sandbox|
  env = { "LLM_TRAINER_ROOT" => sandbox, "TOKENIZER_MODEL_DIR" => "" }
  out = IO.popen([env, RbConfig.ruby, File.join(SCRIPTS, "bin", "train"),
                  "--restore", "_mlx/nope", "--model", MODEL], err: [:child, :out], &:read)
  status = $?
  assert !status.success?, "bin/train --restore exits non-zero for a missing snapshot"
  assert out.include?("Snapshot not found"), "error names the missing snapshot"
  assert !out.include?("Errno"), "no raw exception leaks to the user"
end

# --- auto-split counts for small datasets ---
assert_equal [164, 41], auto_split_counts(205), "~20% held out for validation"
assert_equal [80, 20], auto_split_counts(100), "rounds to 20%"
assert_equal [10, 10], auto_split_counts(20), "at least 10 valid when the set allows"
assert_equal [6, 6], auto_split_counts(12), "never more than half for tiny sets"
assert_equal [2, 1], auto_split_counts(3), "tiny set still leaves some training data"
assert_equal [1, 0], auto_split_counts(1), "single entry: no validation possible"
# (auto_split_counts is only called for small sets; large sets keep the
# requested slice untouched — covered by the defaults test above)

# --- auto-split applied by bin/train on a small dataset (dry-run) ---
Dir.mktmpdir("small_sft_sandbox") do |sandbox|
  dataset = File.join(sandbox, "_dataset")
  FileUtils.mkdir_p(dataset)
  File.write(File.join(dataset, "sft_train_set.jsonl"),
             (["{}\n"] * 205).join)
  env = { "LLM_TRAINER_ROOT" => sandbox, "TOKENIZER_MODEL_DIR" => "" }
  out = IO.popen([env, RbConfig.ruby, File.join(SCRIPTS, "bin", "train"),
                  "--dry-run", "--model", MODEL], err: [:child, :out], &:read)
  status = $?
  assert status.success?, "bin/train --dry-run succeeds on a small dataset"
  assert out.include?("Auto-split: 164 train / 41 valid"), "auto-split announced with the computed counts"
  assert out.include?("164") && out.include?("41"), "build command carries the auto-split counts"
end

# --- trained-but-unexported guard: plain run refuses; --force and --resume pass ---
Dir.mktmpdir("guard_sandbox") do |sandbox|
  dataset = File.join(sandbox, "_dataset")
  adapter_dir = File.join(sandbox, "_mlx", "adapters_qwen3")
  FileUtils.mkdir_p(dataset)
  FileUtils.mkdir_p(adapter_dir)
  File.write(File.join(dataset, "sft_train_set.jsonl"), "{}\n")
  File.write(File.join(adapter_dir, "adapters.safetensors"), "trained-weights")
  env = { "LLM_TRAINER_ROOT" => sandbox, "TOKENIZER_MODEL_DIR" => "" }

  out = IO.popen([env, RbConfig.ruby, File.join(SCRIPTS, "bin", "train"),
                  "--dry-run", "--model", MODEL], err: [:child, :out], &:read)
  assert !$?.success?, "plain bin/train refuses to run over trained state"
  assert out.include?("Trained model state found"), "error names the trained state"
  assert out.include?("--resume"), "error offers --resume"
  assert out.include?("--force"), "error offers --force"
  assert !out.include?("Errno"), "no raw exception leaks to the user"

  ok = system(env, RbConfig.ruby, File.join(SCRIPTS, "bin", "train"),
              "--dry-run", "--force", "--model", MODEL, out: File::NULL, err: File::NULL)
  assert ok, "bin/train --force --dry-run succeeds over trained state"
end

# --- clear_adapter_dir removes the adapter dir ---
Dir.mktmpdir("clear_sandbox") do |d|
  adapter = File.join(d, "adapters_qwen3")
  FileUtils.mkdir_p(adapter)
  File.write(File.join(adapter, "adapters.safetensors"), "x")
  clear_adapter_dir(adapter)
  assert !File.exist?(adapter), "clear_adapter_dir removes the adapter dir"
end

# --- last_session_val_summary: baseline/best/final of the most recent session ---
Dir.mktmpdir("log_parse") do |d|
  log = File.join(d, "train.log")
  File.write(log, <<~LOG)
    Loading pretrained model
    Starting training..., iters: 100
    Iter 1: Val loss 0.906, Val took 51.522s
    Iter 20: Train loss 1.176, Learning Rate 1.000e-05
    Iter 100: Val loss 0.611, Val took 51.294s
    Starting training..., iters: 2183
    Iter 1: Val loss 1.124, Val took 21.711s
    Iter 200: Val loss 0.955, Val took 18.337s
    Iter 1400: Val loss 0.889, Val took 11.652s
    Iter 2000: Val loss 0.643, Val took 18.316s
    Iter 2183: Val loss 0.977, Val took 14.556s
  LOG
  s = last_session_val_summary(log)
  assert_equal 1.124, s[:baseline_val], "baseline is the first eval of the last session"
  assert_equal 0.643, s[:best_val], "best picks the lowest val of the last session"
  assert_equal 2000, s[:best_iter], "best iter reported alongside the best val"
  assert_equal 0.977, s[:final_val], "final is the last eval of the last session"
  assert_equal 2183, s[:final_iter], "final iter reported alongside the final val"
  assert_nil last_session_val_summary(File.join(d, "missing.log")), "missing log → nil"
end

# --- last_session_val_summary edge cases ---
Dir.mktmpdir("log_parse_edges") do |d|
  File.write(File.join(d, "no_marker.log"), <<~LOG)
    Iter 1: Val loss 0.906, Val took 51.522s
    Iter 100: Val loss 0.611, Val took 51.294s
  LOG
  assert_nil last_session_val_summary(File.join(d, "no_marker.log")), "no session marker → nil"

  File.write(File.join(d, "no_evals.log"), <<~LOG)
    Starting training..., iters: 100
    Iter 20: Train loss 1.176, Learning Rate 1.000e-05
  LOG
  assert_nil last_session_val_summary(File.join(d, "no_evals.log")), "session marker without evals → nil"
end

# --- assess_session_summary: within-session learned + drift verdicts ---
lines = assess_session_summary(baseline_val: 0.906, best_val: 0.611, best_iter: 100,
                               final_val: 0.611, final_iter: 100)
assert lines[0].start_with?("learned well"), "clear improvement verdict"
assert lines[0].include?("0.906") && lines[0].include?("0.611"), "improvement shows start and best"
assert lines[0].include?("iter 100"), "improvement shows the best iteration"
assert lines[1].start_with?("stable"), "final matching the best → stable"

lines = assess_session_summary(baseline_val: 1.124, best_val: 0.643, best_iter: 2000,
                               final_val: 0.977, final_iter: 2183)
assert lines[0].start_with?("learned well"), "learned despite late drift"
assert lines[1].start_with?("late-run drift"), "worse final → drift verdict"
assert lines[1].include?("--checkpoint 2000"), "drift advice names the val-best checkpoint"

lines = assess_session_summary(baseline_val: 1.000, best_val: 1.100, best_iter: 50,
                               final_val: 1.100, final_iter: 100)
assert lines[0].start_with?("little learning"), "regression verdict"

lines = assess_session_summary(baseline_val: 1.000, best_val: 0.980, best_iter: 50,
                               final_val: 0.980, final_iter: 100)
assert lines[0].start_with?("modest change"), "small change verdict"

# --- resume_from_val_best: drift → resume from the val-best checkpoint ---
Dir.mktmpdir("resume_best") do |d|
  adapter_dir = File.join(d, "_mlx", "adapters_qwen3")
  FileUtils.mkdir_p(adapter_dir)
  File.write(File.join(adapter_dir, "adapters.safetensors"), "drifted-final-weights")
  File.write(File.join(adapter_dir, "0002600_adapters.safetensors"), "val-best-weights")
  File.write(File.join(d, "train_proper.log"), <<~LOG)
    Starting training..., iters: 2816
    Iter 1: Val loss 1.193, Val took 10s
    Iter 2600: Val loss 0.824, Val took 10s
    Iter 2816: Val loss 1.173, Val took 10s
  LOG

  msg = resume_from_val_best(File.join(d, "train_proper.log"), adapter_dir)
  assert msg, "drift detected → a message is returned"
  assert msg.include?("iter 2600") && msg.include?("0.824"), "message names the val-best"
  assert msg.include?("1.173"), "message names the drifted final"
  assert_equal "val-best-weights",
               File.read(File.join(adapter_dir, "adapters.safetensors")),
               "adapters.safetensors replaced with the val-best checkpoint"
end

# --- resume_from_val_best: no-op cases ---
Dir.mktmpdir("resume_best_noop") do |d|
  adapter_dir = File.join(d, "_mlx", "adapters_qwen3")
  FileUtils.mkdir_p(adapter_dir)
  File.write(File.join(adapter_dir, "adapters.safetensors"), "live-weights")

  # no log at all
  assert_nil resume_from_val_best(File.join(d, "missing.log"), adapter_dir),
             "missing log → no change"

  # log exists but the session never saved a checkpoint for its best (iter 1)
  File.write(File.join(d, "train_proper.log"), <<~LOG)
    Starting training..., iters: 100
    Iter 1: Val loss 1.0, Val took 10s
    Iter 100: Val loss 1.3, Val took 10s
  LOG
  assert_nil resume_from_val_best(File.join(d, "train_proper.log"), adapter_dir),
             "best at iter 1 (no checkpoint saved) → no change"

  # final IS the best (no drift) — identical files
  File.write(File.join(d, "train_proper.log"), <<~LOG)
    Starting training..., iters: 346
    Iter 1: Val loss 1.334, Val took 10s
    Iter 346: Val loss 0.890, Val took 10s
  LOG
  File.write(File.join(adapter_dir, "0000346_adapters.safetensors"), "live-weights")
  assert_nil resume_from_val_best(File.join(d, "train_proper.log"), adapter_dir),
             "final already the best → no change"

  # checkpoint file for the best iter was deleted
  File.write(File.join(d, "train_proper.log"), <<~LOG)
    Starting training..., iters: 100
    Iter 1: Val loss 1.2, Val took 10s
    Iter 100: Val loss 0.9, Val took 10s
  LOG
  assert_nil resume_from_val_best(File.join(d, "train_proper.log"), adapter_dir),
             "missing checkpoint file → no change"
end

# --- run_with_log tees output to a file ---
Dir.mktmpdir("log_sandbox") do |d|
  log = File.join(d, "train.log")
  ok = run_with_log(["printf", "line one\nline two\n"], log)
  assert ok, "run_with_log succeeds"
  assert_equal "line one\nline two\n", File.read(log), "run_with_log tees all output to the log"
end

# --- run_with_log append mode keeps previous content (resume sessions) ---
Dir.mktmpdir("log_sandbox_append") do |d|
  log = File.join(d, "train.log")
  File.write(log, "old session\n")
  ok = run_with_log(["printf", "new session\n"], log, append: true)
  assert ok, "run_with_log append succeeds"
  assert_equal "old session\nnew session\n", File.read(log), "append keeps previous session's log"
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

# --- --resume with no checkpoint: friendly failure, not a crash ---
Dir.mktmpdir("no_resume_sandbox") do |sandbox|
  dataset = File.join(sandbox, "_dataset")
  FileUtils.mkdir_p(dataset)
  File.write(File.join(dataset, "sft_train_set.jsonl"), "{}\n")
  env = { "LLM_TRAINER_ROOT" => sandbox, "TOKENIZER_MODEL_DIR" => "" }
  out = IO.popen([env, RbConfig.ruby, File.join(SCRIPTS, "bin", "train"),
                  "--dry-run", "--resume", "--model", MODEL], err: [:child, :out], &:read)
  status = $?
  assert !status.success?, "bin/train --resume exits non-zero when no checkpoint exists"
  assert out.include?("No checkpoint to resume from"), "error names the missing checkpoint"
  assert out.include?("adapters.safetensors"), "error names the checkpoint file"
  assert out.include?("ruby bin/train"), "error tells the user to run bin/train first"
  assert !out.include?("Errno"), "no raw exception leaks to the user"
end

# --- --resume with a checkpoint present, dry-run succeeds ---
Dir.mktmpdir("resume_ready_sandbox") do |sandbox|
  dataset = File.join(sandbox, "_dataset")
  adapter_dir = File.join(sandbox, "_mlx", "adapters_qwen3")
  FileUtils.mkdir_p(dataset)
  FileUtils.mkdir_p(adapter_dir)
  File.write(File.join(dataset, "sft_train_set.jsonl"), "{}\n")
  File.write(File.join(adapter_dir, "adapters.safetensors"), "weights")
  env = { "LLM_TRAINER_ROOT" => sandbox, "TOKENIZER_MODEL_DIR" => "" }
  ok = system(env, RbConfig.ruby, File.join(SCRIPTS, "bin", "train"),
              "--dry-run", "--resume", "--model", MODEL, out: File::NULL, err: File::NULL)
  assert ok, "bin/train --resume --dry-run succeeds when a checkpoint exists"
end
