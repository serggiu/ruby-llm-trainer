#!/usr/bin/env ruby
# frozen_string_literal: true

# End-to-end test of the build pipeline (`bin/build`) against a sandbox:
# a temp data root (LLM_TRAINER_ROOT) containing a tiny fake source repo
# with code, tests, guides, a license, and a git history with a real
# bug-fix commit. No model, no MLX, no network needed.

require "rbconfig"
require "tempfile"
require "fileutils"
require "json"
require_relative "helper"

RUBY = RbConfig.ruby
SCRIPTS = File.expand_path("..", __dir__) # this repository

def git_env
  {
    "GIT_AUTHOR_NAME" => "Test Author",
    "GIT_AUTHOR_EMAIL" => "test@example.com",
    "GIT_COMMITTER_NAME" => "Test Author",
    "GIT_COMMITTER_EMAIL" => "test@example.com",
  }
end

# Creates a temp data root with a tiny git repo: lib file, test file,
# guides (documents.yaml layout), a MIT license, and a bug-fix commit.
def make_sandbox
  dir = Dir.mktmpdir("llm_trainer_test")
  repo = File.join(dir, "_sources", "demo_repo")
  FileUtils.mkdir_p(File.join(repo, "lib"))
  FileUtils.mkdir_p(File.join(repo, "test"))
  FileUtils.mkdir_p(File.join(repo, "guides", "source"))

  File.write(File.join(repo, "lib", "demo.rb"), <<~RUBY)
    # A small demo library used by the test sandbox.
    module Demo
      # Doubles the given number.
      def self.double(n)
        n * 2
      end
    end
  RUBY

  File.write(File.join(repo, "test", "demo_test.rb"), <<~RUBY)
    require "test/unit"
    class DemoTest < Test::Unit::TestCase
      test "double returns twice the input" do
        assert_equal 4, Demo.double(2)
      end
    end
  RUBY

  File.write(File.join(repo, "guides", "source", "documents.yaml"), <<~YAML)
    - name: Getting Started
      documents:
        - name: Demo Guide
          url: getting_started
          description: How to use the demo library.
  YAML

  File.write(File.join(repo, "guides", "source", "getting_started.md"),
             "# Demo Guide\n\n## Installation\n\nAdd the gem.\n\n## Usage\n\nCall Demo.double.\n")

  File.write(File.join(repo, "LICENSE"), <<~TEXT)
    MIT License

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    Copyright (c) 2026 Test Author
  TEXT

  Dir.chdir(repo) do
    system("git", "init", "-q") or raise "git init failed"
    system("git", "add", "-A") or raise "git add failed"
    system(git_env, "git", "commit", "-q", "-m", "initial import") or raise "first commit failed"

    # A real single-file bug fix (lib/demo.rb gains a nil guard).
    File.write(File.join(repo, "lib", "demo.rb"), <<~RUBY)
      # A small demo library used by the test sandbox.
      module Demo
        # Doubles the given number (nil-safe).
        def self.double(n)
          n.nil? ? nil : n * 2
        end
      end
    RUBY
    system("git", "add", "-A") or raise "git add failed"
    system(git_env, "git", "commit", "-q", "-m", "fix nil error in Demo.double") or raise "fix commit failed"
  end

  dir
end

sandbox = make_sandbox
begin
  env = { "LLM_TRAINER_ROOT" => sandbox }
  ok = system(env, RUBY, File.join(SCRIPTS, "bin", "build"),
              out: File::NULL, err: File::NULL)
  assert ok, "bin/build completes successfully in the sandbox"

  # --- code context + code dataset ---
  code_md = File.join(sandbox, "code_context", "demo_repo.md")
  code_jsonl = File.join(sandbox, "_dataset", "code_demo_repo.jsonl")
  assert File.exist?(code_md), "code context dump written"
  assert File.read(code_md).include?("module Demo"), "code dump contains the source"
  assert File.exist?(code_jsonl), "code dataset written"
  code_lines = File.readlines(code_jsonl)
  assert code_lines.size >= 2, "code dataset has reproduction + explanation entries"
  first = JSON.parse(code_lines.first)
  assert_equal "human", first["conversations"][0]["from"], "ShareGPT human turn"
  assert_equal "gpt", first["conversations"][1]["from"], "ShareGPT gpt turn"
  assert code_lines.all? { |l| JSON.parse(l)["conversations"].size == 2 },
         "every line is a valid 2-turn ShareGPT conversation"

  # --- docs context + docs dataset ---
  guide = File.join(sandbox, "docs_context", "demo_repo", "getting-started", "getting_started.md")
  docs_jsonl = File.join(sandbox, "_dataset", "docs_demo_repo_getting_started.jsonl")
  assert File.exist?(guide), "guide converted to docs_context"
  assert File.exist?(docs_jsonl), "guide dataset written"

  # --- combined dataset ---
  full = File.join(sandbox, "_dataset", "_full_ruby_dataset.jsonl")
  assert File.exist?(full), "combined dataset written"
  assert File.readlines(full).size >= 3, "combined dataset aggregates code + docs"

  # --- pretrain corpus ---
  corpus = File.join(sandbox, "_pretrain", "ruby_corpus.jsonl")
  assert File.exist?(corpus), "pretrain corpus written"
  assert JSON.parse(File.readlines(corpus).first).key?("text"),
         "corpus lines are {text: ...}"

  # --- attribution ---
  attr = File.join(sandbox, "Attribution.md")
  assert File.exist?(attr), "attribution written"
  assert File.read(attr).include?("demo_repo"), "attribution lists the sandbox repo"
  assert File.read(attr).include?("MIT"), "attribution detects the MIT license"

  # --- sft set: all four pair types ---
  sft = File.join(sandbox, "_dataset", "sft_train_set.jsonl")
  assert File.exist?(sft), "sft set written"
  humans = File.readlines(sft).map { |l| JSON.parse(l)["conversations"][0]["value"] }
  assert humans.any? { |h| h.include?("Demo.double") }, "method-implementation pair present"
  assert humans.any? { |h| h.include?("DemoTest") }, "test → implementation pair present"
  assert humans.any? { |h| h.include?("fix nil error") }, "bug-fix pair mined from git history"
  assert humans.any? { |h| h.include?("Demo Guide") }, "guide how-to pair present"
ensure
  FileUtils.remove_entry(sandbox) if sandbox
end
