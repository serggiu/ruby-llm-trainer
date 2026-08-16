#!/usr/bin/env ruby
# frozen_string_literal: true

# Tests for bin/export's plan building and checkpoint selection (no model, no
# MLX needed — everything runs against fake paths or a sandbox).

require "rbconfig"
require "tempfile"
require "fileutils"
require_relative "helper"

# Loads bin/export as a library: defines SCRIPTS/ROOT/build_plan/
# checkpoint_adapter_dir/run_with_log; the __FILE__ guard keeps its main
# flow from running.
load File.join(File.expand_path("..", __dir__), "bin", "export")

def plan_for(*args, data_root: SCRIPTS)
  build_plan(args, scripts: SCRIPTS, data_root: data_root,
                   model_cache_glob: "/nonexistent/*/tokenizer.json")
end

MODEL = "/models/fake"

# --- defaults ---
p = plan_for("--model", MODEL)
assert_equal "_mlx/adapters_qwen3", p[:adapter], "default adapter"
assert p[:adapter_dir].end_with?("_mlx/adapters_qwen3"), "adapter dir under the data root"
assert_equal "_mlx/model_qwen3_export", p[:out], "default out"
assert p[:out_dir].end_with?("_mlx/model_qwen3_export"), "out dir under the data root"
assert p[:log_path].end_with?("_mlx/export.log"), "log path under the data root"
assert_nil p[:checkpoint], "no checkpoint by default"
assert_equal false, p[:dequantize], "dequantize off by default"
assert_equal false, p[:dry_run], "dry-run off by default"
assert p[:fuse_cmd].any? { |x| x.include?("mlx_lm.fuse") }, "fuse command uses the venv mlx_lm.fuse"
assert p[:fuse_cmd].include?("--save-path") && p[:fuse_cmd].include?(p[:out_dir]), "fuse targets the out dir"
assert !p[:fuse_cmd].include?("--dequantize"), "no --dequantize by default"

# --- flags ---
p2 = plan_for("--model", MODEL, "--adapter", "_mlx/my_adapter", "--checkpoint", "2500",
              "--out", "_mlx/model_test", "--dequantize", "--dry-run")
assert_equal "_mlx/my_adapter", p2[:adapter], "--adapter parsed"
assert_equal 2500, p2[:checkpoint], "--checkpoint parsed"
assert_equal "_mlx/model_test", p2[:out], "--out parsed"
assert_equal true, p2[:dequantize], "--dequantize parsed"
assert_equal true, p2[:dry_run], "--dry-run parsed"
assert p2[:fuse_cmd].include?("--dequantize"), "--dequantize lands in the command"

# --- errors (clear TOKENIZER_MODEL_DIR so the fallbacks are exercised) ---
old_env = ENV["TOKENIZER_MODEL_DIR"]
ENV["TOKENIZER_MODEL_DIR"] = ""
assert_raises(ArgumentError, "unknown option rejected") { plan_for("--bogus") }
assert_raises(ArgumentError, "missing --model value rejected") { plan_for("--model") }
assert_raises(ArgumentError, "non-numeric --checkpoint rejected") { plan_for("--checkpoint", "abc", "--model", MODEL) }

# --- model auto-detection fallback ---
Dir.mktmpdir("fake_model") do |fake_model|
  File.write(File.join(fake_model, "tokenizer.json"), "{}")
  p3 = build_plan([], scripts: SCRIPTS, data_root: SCRIPTS,
                   model_cache_glob: File.join(fake_model, "tokenizer.json"))
  assert_equal fake_model, p3[:model], "model auto-detected from the cache glob"
end
ENV["TOKENIZER_MODEL_DIR"] = old_env

# --- checkpoint selection copies the checkpoint as adapters.safetensors ---
Dir.mktmpdir("export_checkpoint") do |sandbox|
  adapter_dir = File.join(sandbox, "adapters")
  FileUtils.mkdir_p(adapter_dir)
  File.write(File.join(adapter_dir, "adapter_config.json"), "{}")
  File.write(File.join(adapter_dir, "0002500_adapters.safetensors"), "ckpt-2500")
  File.write(File.join(adapter_dir, "0003000_adapters.safetensors"), "ckpt-3000")

  tmp = checkpoint_adapter_dir(adapter_dir, 2500, sandbox)
  assert File.file?(File.join(tmp, "adapters.safetensors")), "checkpoint copied as adapters.safetensors"
  assert_equal "ckpt-2500", File.read(File.join(tmp, "adapters.safetensors")), "right checkpoint content"
  assert File.file?(File.join(tmp, "adapter_config.json")), "adapter config copied along"
  assert !File.exist?(File.join(tmp, "0003000_adapters.safetensors")), "other checkpoints not copied"

  err = assert_raises(ArgumentError, "missing checkpoint rejected") do
    checkpoint_adapter_dir(adapter_dir, 9999, sandbox)
  end
  assert err.message.include?("9999"), "error names the missing checkpoint"
  assert err.message.include?("2500"), "error lists the available checkpoints"
end

# --- missing adapter: friendly failure, not a crash ---
Dir.mktmpdir("empty_sandbox") do |sandbox|
  env = { "LLM_TRAINER_ROOT" => sandbox, "TOKENIZER_MODEL_DIR" => "" }
  out = IO.popen([env, RbConfig.ruby, File.join(SCRIPTS, "bin", "export"),
                  "--dry-run", "--model", MODEL], err: [:child, :out], &:read)
  status = $?
  assert !status.success?, "bin/export exits non-zero when the adapter is missing"
  assert out.include?("Adapter not found"), "error names the missing adapter"
  assert out.include?("adapters_qwen3"), "error names the missing dir"
  assert out.include?("ruby bin/train"), "error tells the user to run bin/train first"
  assert !out.include?("Errno"), "no raw exception leaks to the user"
end

# --- with an adapter present, dry-run succeeds ---
Dir.mktmpdir("ready_sandbox") do |sandbox|
  adapter_dir = File.join(sandbox, "_mlx", "adapters_qwen3")
  FileUtils.mkdir_p(adapter_dir)
  File.write(File.join(adapter_dir, "adapters.safetensors"), "weights")
  env = { "LLM_TRAINER_ROOT" => sandbox, "TOKENIZER_MODEL_DIR" => "" }
  ok = system(env, RbConfig.ruby, File.join(SCRIPTS, "bin", "export"),
              "--dry-run", "--model", MODEL, out: File::NULL, err: File::NULL)
  assert ok, "bin/export --dry-run succeeds when the adapter exists"
end
