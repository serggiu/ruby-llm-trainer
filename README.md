<p align="center">
  <img src="assets/runner.svg" width="560" alt="Ruby Trainer — a runner on the track">
</p>

<p align="center">
  <a href="https://github.com/ml-explore/mlx"><img src="https://img.shields.io/badge/training-MLX%20on%20Apple%20silicon-000000" alt="Trained with MLX"></a>
</p>

# Ruby LLM Trainer — MLX fine-tuning for AI coding agents

A small pipeline that turns any Ruby repositories placed under `_sources/`
into:

1. clean, agent-friendly Markdown context files
2. a training dataset for local LLMs

The generated Markdown files can be used as a Ruby or Ruby on Rails skill for
your local agents. The datasets are plain ShareGPT JSONL, so they work with
any fine-tuning pipeline — **MLX** locally (see [TRAIN.md](TRAIN.md)),
LLaMA-Factory, axolotl, Unsloth, ...

Clone any Ruby repositories (private or open-source) into `_sources/` — the
scripts run `git pull` in each folder (a failed pull only warns and continues
with what's on disk), so you always build from the latest version. Any repo
works; ones without `.rb` files are simply skipped.

## Run everything (`bin/build`)

One command regenerates the whole training corpus — docs context, code
context, datasets, attribution, pretrain corpus, and the SFT (supervised fine-tuning) set:

```bash
ruby bin/build                  # full pipeline
ruby bin/build --skip-bugs      # skip the (slower) git-history bug mining
```

It calls the orchestrator `build/main.rb`, then builds the pretrain corpus
and the SFT set. The steps, in order:

1. `build_docs_context.rb` — Ruby docs and guides
2. `build_code_context.rb` — Ruby source code
3. `create_dataset.rb` — the main dataset
4. `build_attribution.rb` — repo/file attribution
5. `build_pretrain_corpus.rb` — the pretrain corpus
6. `build_sft_pairs.rb` — the SFT pairs

Each script can also be run individually — see below.

When it finishes, the training file is `_dataset/_full_ruby_dataset.jsonl`:
one ShareGPT conversation per line, covering every Ruby source file and
guide in the configured repositories. 

Every pair is capped at
`MAX_CONTEXT_TOKENS` (2048 estimated tokens) — long files and chapters are
split into as many "part i/n" pairs as needed, so nothing is truncated at
training time.

## Train (`bin/train`)

Rebuilds the MLX chat-format slice from the fresh SFT set and runs LoRA
training on it:

```bash
ruby bin/train                              # 1000 iters on the proper slice
ruby bin/train --iters 5000 --slice full    # longer run on the full slice
ruby bin/train --watchdog                   # supervised by the memory watchdog
ruby bin/train --dry-run                    # print the commands, don't run
```

Options:

- `--iters N` — training iterations (default 1000)
- `--slice smoke|proper|full` — data slice (300/50, 3000/200, 20000/700)
- `--adapter DIR` — adapter output dir (default `_mlx/adapters_qwen3`)
- `--watchdog` — supervise the run with the memory watchdog (WATCHDOG=1)
- `--resume` — continue from the latest adapter checkpoint instead of a fresh run
- `--model DIR` — model directory (default: the Qwen3-8B MLX cache)
- `--dry-run` — print the commands without running them

Sessions can be chained with `--resume` (stop, continue later), but a single
long run is usually better — see [Resuming a run](TRAIN.md#resuming-a-run--and-why-one-long-run-beats-several-chained-ones) in [TRAIN.md](TRAIN.md).

Every `bin/train` run starts from the **base model** with **fresh, randomly
initialized LoRA weights** — previous training is never loaded implicitly.
The only way to continue where you left off is `--resume`, which loads the
latest adapter checkpoint (`_mlx/adapters_qwen3/adapters.safetensors`)
before training starts. Note that a fresh run saves checkpoints over the
old ones in the adapter dir — copy the folder first if you want to keep a
previous run's checkpoints.

`bin/train` needs the SFT set produced by the build step — if it's missing
(e.g. you haven't run `bin/build` yet), it fails with a message telling you
to run `ruby bin/build` first.

**To train a model on this data** (install MLX, run continued pretraining +
SFT, evaluate the result), follow [TRAIN.md](TRAIN.md).

Training summary by DeepSeek V4 Flash:
> The model is not a student being quizzed and corrected — it's a parrot being shown thousands of correct transcripts and learning to continue them.

## Check the trained model

Before exporting, decide whether the adapter is good enough. Chat with the base
model and with the adapter on the same prompts, using the same seed so the
outputs are directly comparable:

```bash
# Base model (no adapter):
.venv/bin/mlx_lm.generate \
  --model <model-dir> \
  --seed 42 --max-tokens 300 --use-default-chat-template \
  --prompt "Explain the difference between `include` and `prepend` in Ruby modules?"

# Trained adapter on top of the same model:
.venv/bin/mlx_lm.generate \
  --model <model-dir> \
  --adapter-path _mlx/adapters_qwen3 \
  --seed 42 --max-tokens 300 --use-default-chat-template \
  --prompt "Explain the difference between `include` and `prepend` in Ruby modules?"
```

`<model-dir>` is the model you trained with — by default the Qwen3-8B MLX
cache under `~/.cache/huggingface/hub/`. Try a handful of Ruby questions plus
one general question. The adapter is **good enough** when:

- it answers the Ruby questions correctly and directly — the base model often
  stalls in its thinking mode instead;
- it answers in one pass, without trailing `<think>` rambling;
- it still chats normally on the general question (no regression).

If it's not good enough, train more (`ruby bin/train --iters N` — longer runs
see more of the dataset) before exporting.

## Export the trained model (`bin/export`)

Once the adapter is good enough, export it as a new standalone model — the
result needs no base snapshot afterwards:

```bash
ruby bin/export                       # export the latest adapter
ruby bin/export --checkpoint 2500     # export a specific training checkpoint
ruby bin/export --dequantize          # export fp16 instead of keeping 4-bit
ruby bin/export --dry-run             # print the commands, don't run
```

Options:

- `--adapter DIR` — adapter dir to export (default `_mlx/adapters_qwen3`)
- `--checkpoint N` — export checkpoint N instead of the latest adapter
- `--out DIR` — output dir (default `_mlx/model_qwen3_export`)
- `--model DIR` — model directory (default: the Qwen3-8B MLX cache)
- `--dequantize` — keep the fp16 export instead of the 4-bit default
- `--dry-run` — print the commands without running them

The default export is **4-bit and correct**: the adapter is fused into an
fp16 model first, then the fused weights are quantized (`mlx_lm.fuse
--dequantize` → `mlx_lm.convert -q`). Measured on Qwen3-8B, the naive
single-step 4-bit fuse re-quantizes inline and loses the adapter entirely —
the export then behaves like the base model. `--dequantize` skips the
quantization step and keeps the fp16 model (roughly double the size,
~16 GB for the 8B). The write is atomic: if the export fails, the previous
exported model is left intact. `bin/export` needs a trained adapter — if
it's missing, it tells you to run `ruby bin/train` first.

## Training recommendation

> **Never remove previous sources when adding new ones** — accumulation
> preserves knowledge breadth — and **always do the final training as one
> combined run** over the accumulated set, not as a sequence of incremental
> checkpoints.

`bin/build` regenerates the combined dataset from **every** repository under
`_sources/`, so adding a source automatically includes it alongside all
previous ones. Incremental runs (train on a subset, evaluate, add a source,
repeat) are still useful as a **measurement tool** — they tell you which
sources actually improve the model on a fixed eval set. Use them to decide
what stays in `_sources/`, then train the final model on the whole
accumulated dataset in a single run (see [TRAIN.md](TRAIN.md)).

---

## Prerequisites

- **Ruby 3+** — the scripts use only the standard library; the whole data
  pipeline (generation, tokenization, verification) is pure Ruby. Python is
  used only by the optional MLX training environment (see [TRAIN.md](TRAIN.md)).
- Install Ruby with [rbenv](https://github.com/rbenv/rbenv) or another Ruby
  package manager: `ruby --version` (expect 3.x or newer).
- Git clones of any Ruby repositories under `_sources/` — the build scripts
  `git pull` each repo before building, so the clones only need to exist once.

---

## Build Scripts

All scripts are **source-agnostic**: nothing is hardcoded to specific
repositories — whatever is under `_sources/` is picked up automatically, and
outputs for removed repos are cleaned up.

### `build_code_context.rb` — repos → `code_context/` + `_dataset/code_*.jsonl`

Walks every `.rb` file in every repository and produces one agent-friendly
Markdown dump and one ShareGPT dataset per repo: code reproduction, API
explanation (from the real RDoc comments), and test coverage listings.

```bash
ruby build/build_code_context.rb [sources_dir]
```

### `build_docs_context.rb` — repos with guides → `docs_context/`

Converts any repo shipping a `guides/source/documents.yaml` layout into
`docs_context/<repo>/<category>/<guide>.md` (YAML front matter kept); repos
without guides are skipped. For guides following the
`<major>_<minor>_release_notes` naming, only the newest version is kept
(pass `--all-release-notes` to keep every version).

```bash
ruby build/build_docs_context.rb [sources_dir]
```

### `create_dataset.rb` — aggregate everything → `_dataset/`

Converts the guides into one dataset per guide (overview + per-chapter
entries), picks up the `code_*.jsonl` datasets as-is, and concatenates
everything into `_dataset/_full_ruby_dataset.jsonl`, with an `INDEX.md`
manifest. The combined write is atomic and self-validating.

```bash
ruby build/create_dataset.rb [docs_context_dir] [output_dir]
```

### `build_dataset.rb` — one Markdown dump → one dataset file

The original single-file converter behind `build_code_context.rb`: turns one
code-context Markdown dump into a ShareGPT JSONL dataset (code reproduction,
API explanation, test coverage).

**Context-length bounding**: every generated pair is capped at
`MAX_CONTEXT_TOKENS` (2048 estimated tokens) — answers longer than the cap
(big source files, long guide chapters) are split into as many
`(prompt, part i/n)` pairs as needed. The same rule applies to the code,
docs, and SFT entry builders.

```bash
ruby build/build_dataset.rb [input.md] [output.jsonl]
```

### `build_pretrain_corpus.rb` — raw material → pretraining corpus

Builds the continued-pretraining corpus: every `.rb` file (verbatim) plus
every guide chapter (front matter stripped, split at chapter level), as one
`{"text": ...}` JSON object per line — the format LLaMA-Factory's `pt` stage
and mlx-lm accept. Duplicates are removed by content hash.

```bash
ruby build/build_pretrain_corpus.rb    # → _pretrain/ruby_corpus.jsonl
```

### `build_sft_pairs.rb` — task-oriented SFT (supervised fine-tuning) set

Builds the task-oriented SFT set on top of the existing datasets:

| Source | Content |
|---|---|
| base entries | every `_dataset/*.jsonl` entry, deduplicated |
| method pairs | “implement this method from its API documentation” — RDoc + signature as prompt, real method body as answer |
| test pairs | “implement the class under test” — test file as prompt, matched lib/app file as answer |
| bug-fix pairs | “fix this bug” — real single-file fixes mined from the git history of every repository under `_sources/` |
| guide pairs | how-to Q&A from the guides with diversified natural-language prompts |

```bash
ruby build/build_sft_pairs.rb [--skip-bugs]   # → _dataset/sft_train_set.jsonl
```

### `build_attribution.rb` — sources → `Attribution.md`

Reads each repository's license file, detects the license type and copyright
holder, and writes the attribution table to `Attribution.md` (git-ignored —
it lists the repositories you train with locally). Repos without a license
file are skipped with a warning.

```bash
ruby build/build_attribution.rb    # → Attribution.md
```

---

## Tools

### `tools/tokenizer.rb` — our own tokenizer (pure Ruby)

A self-contained re-implementation of the HuggingFace byte-level BPE
tokenizer (the `tokenizer.json` format): loads a model's tokenizer files and
reproduces the original tokenizer's token IDs exactly for this pipeline's
chat data — verified against the original Python/transformers tokenizer
(identical token IDs on sampled data, 0 count mismatches on 3,000 real
entries). No Python and no gems: standard library only. See
[TOKENIZER.md](tools/TOKENIZER.md) for a step-by-step comparison against the
Python tokenizer.

```bash
ruby tools/tokenizer.rb <model_dir> [--ids] [--chat '<messages json>']
```

As a library:

```ruby
tok = Tokenizer.new(model_dir)
tok.count(text)                # token count of raw text
tok.encode(text)               # array of token ids
tok.apply_chat_template(msgs)  # Qwen-style chat text
tok.count_messages(msgs)       # token count of the templated chat
```

### `tools/seqlen_stats.rb` — dataset length distribution

Reports sample count, min/mean/p50/p90/p99/max token lengths, and how many
samples exceed a threshold (i.e. would be truncated during training). With a
model directory it uses the real tokenizer; without one, the generator's
conservative estimator:

```bash
ruby tools/seqlen_stats.rb dataset.jsonl [model_dir] [max_len]
```

### `tools/find_long.rb` — find over-threshold entries

Lists the entries of a dataset whose token length exceeds a threshold (same
two counting modes as `seqlen_stats.rb`):

```bash
ruby tools/find_long.rb dataset.jsonl [model_dir] [max_len] [limit]
```

### `tools/memmon.rb` — RAM monitor

Polls system-wide free memory every 2 s and prints timestamped readings —
run it in the background while training, then read the log:

```bash
ruby tools/memmon.rb [duration_seconds]   # default 1800
```

### `tools/run_capped.rb` — optional training watchdog

Wraps a training command. By default it is a transparent pass-through
(signals and exit codes behave as if the command ran directly); with
`WATCHDOG=1` it polls free RAM and SIGKILLs the run if it drops below
`--min-free-gb` (default 4 GB) — an abort instead of a system freeze:

```bash
WATCHDOG=1 ruby tools/run_capped.rb --min-free-gb 4 -- .venv/bin/mlx_lm.lora ...
```

## Tests

```bash
ruby bin/test            # runs every test/test_*.rb; exits non-zero on failure
```

Six test files cover the splitting logic, the tokenizer (mechanics always,
golden token-ID tests when a model directory is available — auto-detected
from the Qwen3-8B MLX cache, or `ruby bin/test --model DIR`), the two
diagnostic tools, and the build/train/export commands (run against an in-memory
sandbox — no model or network needed).

---

## How agents use the context files

### Source code context (`code_context/`)

```
"Read code_context/INDEX.md to see what components are available, then load
code_context/<repo>.md before fixing this query bug."
```

### Documentation context (`docs_context/`)

```
"Read docs_context/INDEX.md for available guides. Load
docs_context/<repo>/<category>/<guide>.md for reference on the API."
```

### Combine with project conventions

Always pair with `AGENTS.md` (in the source repository root) for coding
conventions, test commands, and architectural guidance.

---

## File structure of this directory

```
~/Desktop/ruby-trainer/
├── README.md                   ← You are here
├── TRAIN.md                    ← Step-by-step local model training (MLX)
├── RESULTS.md                  ← Log of training runs (hardware, time, loss)
├── bin/build                   ← One-shot data pipeline (docs + code + datasets + SFT)
├── bin/train                   ← Rebuild slice + run LoRA training
├── bin/export                   ← Export the trained adapter as a standalone model
├── bin/test                    ← Runs all tests (test/test_*.rb)
├── build/                      ← Pipeline scripts: main.rb + build_*.rb + create_dataset.rb
├── test/                       ← Tests: split logic + tokenizer golden values
├── tools/                      ← tokenizer.rb + TOKENIZER.md + diagnostic helpers (seqlen_stats, find_long, memmon, run_capped)
├── _sources/                   ← Git clones of the source repositories
├── Attribution.md              ← Generated (git-ignored): license table per source
├── code_context/               ← Generated: one .md + one dataset per repo
├── docs_context/               ← Generated: guides per repo
├── _pretrain/                  ← Generated: ruby_corpus.jsonl (CPT data)
├── _dataset/                   ← Generated: datasets + _full_ruby_dataset.jsonl
└── _mlx/                       ← MLX artifacts (train/valid data, adapters)
```

---

## Licensing

This repository itself is **MIT licensed** — see [LICENSE](LICENSE).

The tokenizer (`tools/tokenizer.rb`) is an original Ruby implementation with
no copied HuggingFace code; its byte table and pre-tokenization regex are
functionally equivalent to OpenAI's GPT-2 (MIT), and its vocabulary/merge
data is loaded at runtime from the model's own `tokenizer.json` (Qwen3:
Apache-2.0) — see [TOKENIZER.md](tools/TOKENIZER.md).

The context files and datasets can be derived from third-party open-source
projects, each with its own license. Every repository under `_sources/` keeps
its license file; each derived dataset carries the license of the repository
it came from.

The per-source attribution table (project, license, copyright holder) is
generated into `Attribution.md` by `build_attribution.rb` — see that file.
It is git-ignored, since it lists the exact repositories used for local
training; regenerate it anytime with:

```bash
ruby build/build_attribution.rb
```

All licenses in use permit the intended use of this project: training and
running a local LLM from `_dataset/_full_ruby_dataset.jsonl`. Repositories
that ship guides may license the guides separately from the code (share-alike
clauses such as CC BY-SA apply) — check the repository. Before **publishing
or serving** a trained model, review each source repository's license
(share-alike and non-compete clauses may apply).

---

## Acknowledgements

The training flow in this repository runs on **MLX** and **MLX-LM**, Apple's
machine learning frameworks for Apple silicon:

- [ml-explore/mlx](https://github.com/ml-explore/mlx) — MIT licensed,
  Copyright © 2023 Apple Inc.
- [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) — MIT licensed,
  Copyright © 2023 Apple Inc.
