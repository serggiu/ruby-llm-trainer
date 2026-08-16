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
your local agents.

If you want to go one step further, the datasets can be used to train local
LLMs — on Apple's **MLX** framework (see [TRAIN.md](TRAIN.md)). The datasets
are plain ShareGPT JSONL, so they also work with any other fine-tuning
pipeline (LLaMA-Factory, axolotl, Unsloth, ...).

As source code, you can use private repositories or any open-source Ruby
libraries you consider to be good quality code (for example MIT-licensed).
Clone them as git repositories into `_sources/` — the scripts run `git pull`
in each folder so you always build from the latest version (a failed pull is
only a warning).

Any repo works; ones without `.rb` files are simply skipped.

## Run everything (main.rb)

The one-shot pipeline. It pulls the latest sources, rebuilds the code and
docs context, and regenerates the whole training corpus:

```bash
ruby main.rb
```

That runs, in order:

1. `build_docs_context.rb` — `git pull` every repo under `_sources/`, rebuild `docs_context/`
2. `build_code_context.rb` — `git pull` every repo under `_sources/`, rebuild `code_context/`
3. `create_dataset.rb` — rebuild every dataset and concatenate them all

When it finishes, the final training file is:

```
_dataset/_full_ruby_dataset.jsonl
```

One ShareGPT-format conversation per line
(`{"conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}`), accepted
out of the box by local fine-tuning pipelines such as **LLaMA-Factory**,
**axolotl**, **Unsloth**, and `llama.cpp`'s finetune tooling — covering every
Ruby source file and guide in the configured repositories. Every pair is
capped at `MAX_CONTEXT_TOKENS` (2048 estimated tokens) — long files and
chapters are split into as many "part i/n" pairs as needed, so nothing is
truncated at training time.

Each step can also be run individually — see the sections below.

**To train a model on this data** (install MLX, run continued pretraining +
SFT, evaluate the result), follow [TRAIN.md](TRAIN.md).

## Training recommendation

A simple rule of thumb for building up the training data over time:

> **Never remove previous sources when adding new ones** — accumulation
> preserves knowledge breadth — and **always do the final training as one
> combined run** over the accumulated set, not as a sequence of incremental
> checkpoints.

Since `main.rb` regenerates `_full_ruby_dataset.jsonl` from **every**
repository under `_sources/`, adding a new source automatically includes it
alongside all previous ones — the dataset always accumulates. One combined
training run over that full set gives the most stable, balanced result.

Incremental runs (train on a subset, evaluate, add a source, repeat) are
still useful, but as a **measurement tool**: they tell you which sources
actually improve the model on a fixed eval set. Use them to decide what stays
in `_sources/`, then train the final model on the whole accumulated dataset
in a single run (see [TRAIN.md](TRAIN.md)).

---

## Prerequisites

- **Ruby 3+** (the scripts use only the standard library).
- Install a modern Ruby with
  [rbenv](https://github.com/rbenv/rbenv) or another Ruby package manager:
  ```bash
  ruby --version   # expect 3.x or newer
  ```
- Clones of any Ruby repositories under `_sources/` (relative to this project).
  The build scripts `git pull` each repo before building, so the clones only
  need to exist once.

---

## Scripts

### `main.rb` — run the whole pipeline

Orchestrates the three steps above in order and stops if any of them fails.
Prints start time and total elapsed time.

```bash
ruby main.rb
```

### `build_code_context.rb` — repos → `code_context/` + `_dataset/code_*.jsonl`

Fully **source-agnostic**: scans every repository under `_sources/`, runs
`git pull` inside each (a failed pull only warns and continues), walks all
`.rb` files — no assumptions about folder structure — and produces one
Markdown dump plus one ShareGPT dataset per repository:

| Output | Content |
|---|---|
| `code_context/<repo>.md` | agent-friendly source dump (file index + verbatim code) |
| `_dataset/code_<repo>.jsonl` | training dataset: code reproduction, API explanation, test coverage (long entries split into ≤2048-token parts) |

Add any new repo to `_sources/` and re-run — it's picked up automatically;
outputs for repos that are removed are cleaned up.

```bash
ruby build_code_context.rb                       # pulls + processes every repo in _sources/
ruby build_code_context.rb /path/to/sources-dir  # use another sources directory
```

Output goes to `./code_context/` (INDEX.md + one file per repo) and one
`code_*.jsonl` per repo in `./_dataset/`:

```
code_context/
├── INDEX.md              # Master index with file counts and sizes
└── <repo>.md             # All Ruby source from _sources/<repo> (one per repo)
```

If a `git pull` fails (e.g. offline), the build continues with the sources
already on disk, printing a warning.

### `build_docs_context.rb` — repos with guides → `docs_context/`

Source-agnostic, like `build_code_context.rb`: scans every repository under
`_sources/`, runs `git pull` inside each (a failed pull only warns), and
converts any repo that ships a `guides/source/documents.yaml` layout. Repos
without guides are skipped. Output is namespaced per repository:

```bash
ruby build_docs_context.rb                       # pulls + converts every repo with guides
ruby build_docs_context.rb /path/to/sources-dir  # use another sources directory
```

Output goes to `./docs_context/`:

```
docs_context/
├── INDEX.md
└── <repo>/
    ├── <category>/<guide>.md
    └── ... (one folder per repo, one file per guide)
```

Each guide keeps its YAML front matter (title, category, description,
work-in-progress flag). If `git pull` fails (e.g. offline), the build
continues with the sources already on disk, printing a warning.

**Release notes**: for guides following the `<major>_<minor>_release_notes`
naming, only the newest version is kept; older ones are skipped and cleaned
up. Pass `--all-release-notes` to keep every version.

### `create_dataset.rb` — aggregate everything → `_dataset/`

Generates the docs datasets and aggregates the full corpus. The code datasets
(`code_*.jsonl`, produced by `build_code_context.rb`) are picked up as-is;
the guides in `docs_context/` are converted into one dataset per guide
(overview entry + per-chapter entries); and everything is concatenated into
the final `_dataset/_full_ruby_dataset.jsonl`.

```bash
ruby create_dataset.rb                    # uses ./docs_context, ./_dataset
ruby create_dataset.rb docs/ out/         # explicit directories
```

Output goes to `./_dataset/`:

```
_dataset/
├── INDEX.md                  # Manifest: dataset, source, entries, size
├── code_<repo>.jsonl         # From build_code_context.rb (one per repo)
├── docs_<repo>_<guide>.jsonl # One per guide
├── ... (one dataset per repo + per guide)
└── _full_ruby_dataset.jsonl  # Everything concatenated — the training file
```

### `build_dataset.rb` — one Markdown dump → one dataset file

The original single-file converter: turns one code-context Markdown dump into
a ShareGPT-format JSONL dataset (code reproduction, API explanation, test
coverage). `build_code_context.rb` reuses its entry builders.

**Context-length bounding**: every generated pair is capped at
`MAX_CONTEXT_TOKENS` (2048, estimated at ~4 chars/token, including chat-
template overhead) — answers longer than the cap (big source files, long
guide chapters) are split into as many `(prompt, part i/n)` pairs as needed.
This keeps the training data free of over-long contexts: nothing gets
silently truncated at training time, and single-sample memory stays bounded
(~14 GB peak for an 8B 4-bit model on MLX at 2048 tokens). The same rule
applies to the code, docs, and SFT entry builders, and the cap is a single
tunable constant.

```bash
ruby build_dataset.rb                    # first dump found in code_context/
ruby build_dataset.rb path/to/other.md path/to/out.jsonl
```

### `build_pretrain_corpus.rb` — raw material → pretraining corpus

Builds a plain-text corpus for **domain adaptation (continued pretraining)**:
one `{"text": ...}` JSON object per line (the format LLaMA-Factory's `pt`
stage and mlx-lm accept), containing every `.rb` file under `_sources/`
(verbatim) plus every guide chapter (front matter stripped, split at chapter
level). Duplicates are removed by content hash.

```bash
ruby build_pretrain_corpus.rb            # → _pretrain/ruby_corpus.jsonl
```

### `build_sft_pairs.rb` — task-oriented SFT (supervised fine-tuning) set

Builds the **task-oriented SFT training set** on top of the existing datasets:

| Source | Content |
|---|---|
| base entries | every `_dataset/*.jsonl` entry, deduplicated |
| method pairs | “implement this method from its API documentation” — RDoc + signature as prompt, real method body as answer |
| test pairs | “implement the class under test” — test file as prompt, matched lib/app file as answer |
| bug-fix pairs | “fix this bug” — real single-file fixes mined from the git history of every repository under `_sources/` |
| guide pairs | how-to Q&A from the guides with diversified natural-language prompts |

```bash
ruby build_sft_pairs.rb                  # → _dataset/sft_train_set.jsonl
ruby build_sft_pairs.rb --skip-bugs      # skip the (slower) git-history mining
```

### `build_attribution.rb` — sources → `Attribution.md`

Reads the license file of every repository under `_sources/`, detects the
license type and copyright holder from its text, and writes the attribution
table to `Attribution.md` (git-ignored — it lists the repositories you train
with locally). Repos without a license file are skipped with a warning. Fully
source-agnostic — the file always matches whatever repositories are present.

```bash
ruby build_attribution.rb                  # → Attribution.md
```

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
├── main.rb                     ← One-shot pipeline (code + docs + datasets)
├── build_code_context.rb       ← Repos in _sources/ → code_context/ + code datasets
├── build_docs_context.rb       ← Repos with guides → docs_context/
├── build_dataset.rb            ← Single-dump dataset converter
├── create_dataset.rb           ← Aggregates code datasets + guides → _full_ruby_dataset.jsonl
├── build_pretrain_corpus.rb    ← Continued-pretraining corpus builder
├── build_sft_pairs.rb          ← Task-oriented SFT set builder
├── build_attribution.rb        ← Regenerates Attribution.md (git-ignored)
├── build_mlx_data.rb           ← ShareGPT → MLX chat format converter
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

The context files and datasets can be derived from third-party open-source
projects, each with its own license. Every repository under `_sources/` keeps
its license file; each derived dataset carries the license of the repository
it came from.

The per-source attribution table (project, license, copyright holder) is
generated into `Attribution.md` by `build_attribution.rb` — see that file.
It is git-ignored, since it lists the exact repositories used for local
training; regenerate it anytime with:

```bash
ruby build_attribution.rb
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
