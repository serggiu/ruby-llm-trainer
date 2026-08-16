# TRAIN.md — Training a local LLM on the generated dataset (MLX)

This file walks you from a fresh Mac to a locally trained model, using the
dataset this project generates. Everything runs **on your machine** — no
cloud, no API keys. Only the initial installs and model download need network
access.

The training approach (see `README.md` for the reasoning):

1. **Domain adaptation (continued pretraining, optional)** — teach the model
   the raw material: every `.rb` file under `_sources/` plus the guides, as
   plain text.
2. **Task SFT (supervised fine-tuning)** — teach it to *answer*: the full
   conversation dataset plus task pairs (implement-from-docs, test-driven,
   bug fixes, guide Q&A).

Verified with **Qwen/Qwen2.5-3B-Instruct** on Apple Silicon (arm64), 32 GB
RAM. The same commands work for any mlx-compatible model (e.g. your target
9B model — check its HuggingFace page for the architecture/chat template
first).

---

## 1. What needs to be installed (one-time)

### 1.1 Prerequisites

| Requirement | Check |
|---|---|
| macOS on Apple Silicon (arm64) | `uname -m` → `arm64` |
| Ruby 3+ (for the data scripts) | `ruby --version` |
| Python 3.9–3.12 | `python3 --version` |
| ~16 GB free RAM for a 3B model, ~32 GB+ for a 9B | — |

### 1.2 Python virtual environment + MLX

```bash
# from the project root
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install mlx-lm        # installs mlx, mlx-lm, transformers
```

### 1.3 Model weights

Download once (network needed); afterwards everything runs from the cache:

```bash
.venv/bin/pip install "huggingface_hub[cli]"
.venv/bin/huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

(You can also skip this — the first `mlx_lm.lora`/`mlx_lm.generate` call
downloads the model automatically to `~/.cache/huggingface/hub/`.)

### 1.4 Verify the install

```bash
.venv/bin/python -c "import importlib.metadata as md; print(md.version('mlx'), md.version('mlx-lm'))"
.venv/bin/python -c "from mlx_lm import load; m, t = load('Qwen/Qwen2.5-3B-Instruct'); print('model loaded OK')"
```

Expected: two version numbers, then `model loaded OK` (takes ~30 s, ~6 GB RAM).

---

## 2. The data pipeline (build once)

The Ruby scripts (no Python needed) generate everything the trainer consumes:

```bash
# Full context + datasets (pulls _sources/*, builds code_context/, docs_context/, _dataset/)
ruby bin/build

# OR individually:
ruby build/build_code_context.rb     # code_context/ (one file per source repository)
ruby build/build_docs_context.rb     # docs_context/ (guides from every repository)
ruby create_dataset.rb         # _dataset/*.jsonl + _dataset/_full_ruby_dataset.jsonl

# Training data:
ruby build/build_code_context.rb      # code_context/<repo>.md + _dataset/code_<repo>.jsonl
ruby build/build_pretrain_corpus.rb   # _pretrain/ruby_corpus.jsonl  (CPT corpus, {"text": ...})
ruby build/build_sft_pairs.rb         # _dataset/sft_train_set.jsonl (task SFT set, ShareGPT)
ruby build/build_mlx_data.rb 12000 1000   # _mlx/train.jsonl + _mlx/valid.jsonl (MLX chat format)
```

All generated conversation pairs are capped at ~2048 tokens
(`MAX_CONTEXT_TOKENS` in `build_dataset.rb`): long source files and guide
chapters are split into as many `(part i/n)` pairs as needed, so nothing is
truncated during training and memory stays bounded (~13-14 GB peak for an 8B
4-bit model). Use `--max-seq-length 2048` (more only if you add unsplit
data).

`build_mlx_data.rb 12000 1000` writes `_mlx/train.jsonl` + `_mlx/valid.jsonl`
and overwrites them each run. To keep several slices at once (smoke / test /
full), pass the input file and output dir explicitly:
`ruby build/build_mlx_data.rb 3000 200 _dataset/sft_train_set.jsonl _mlx/qwen3_proper`

| File | Format | Used by |
|---|---|---|
| `_pretrain/ruby_corpus.jsonl` | `{"text": "..."}` per line | Stage 1 (CPT) — accepted by MLX as-is |
| `_dataset/sft_train_set.jsonl` | ShareGPT `conversations` | — (intermediate) |
| `_mlx/train.jsonl`, `_mlx/valid.jsonl` | `messages` (chat) | Stage 2 (SFT) |

---

## 3. Training stages

> All commands use the venv binaries (`.venv/bin/...`). Adapters (LoRA
> weights, a few MB) are saved under `_mlx/`; the base model is never modified.

### Stage 1 — Continued pretraining (domain adaptation, optional)

```bash
.venv/bin/mlx_lm.lora \
  --model Qwen/Qwen2.5-3B-Instruct \
  --train \
  --data _pretrain/ruby_corpus.jsonl \
  --adapter-path _mlx/adapters_cpt \
  --iters 1000 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --steps-per-report 50 \
  --steps-per-eval 250 \
  --max-seq-length 2048
```

Notes:

- Recommended on a **base** model. On an already-instructed chat model
  (like Qwen2.5-3B-**Instruct**), use a low learning rate and check for
  regression on general questions (see §5).
- If your mlx-lm version rejects a single JSONL file, split
  `ruby_corpus.jsonl` into `train.jsonl`/`valid.jsonl` in a directory, like
  `_mlx`.
- Expect roughly a few seconds per iteration on a 3B; the smoke run (100
  iters) takes ~10 minutes. The full corpus takes however long the sources
  require — it scales with the number of source files.

### Stage 2 — Task SFT (the main stage)

If you did Stage 1, fuse its adapters into a standalone model first, then
train SFT on top of that; otherwise start from the base model.

```bash
# (only if Stage 1 was run) bake the CPT adapters into a full checkpoint:
.venv/bin/mlx_lm.fuse \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path _mlx/adapters_cpt \
  --save-path _mlx/model_cpt

# SFT on the MLX-format task set:
.venv/bin/mlx_lm.lora \
  --model _mlx/model_cpt \            # or Qwen/Qwen2.5-3B-Instruct (no CPT)
  --train \
  --data _mlx \
  --adapter-path _mlx/adapters_sft \
  --iters 1000 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --steps-per-report 10 \
  --steps-per-eval 100 \
  --max-seq-length 2048
```

Watch the output: `Train loss` and `Val loss` should decrease. The generated
data is pre-split at ~2048 tokens per pair, so with `--max-seq-length 2048`
nothing gets truncated (a truncation warning means the flag is set below the
data's needs, or you're using custom unsplit data). Don't raise the limit
needlessly: on a 4-bit model, mlx-lm's quantized attention materializes O(n²)
attention scores — at 4096 a validation pass spikes to ~40+ GB peak on an 8B,
while 2048 stays at ~13-14 GB.

#### Resuming a run — and why one long run beats several chained ones

Training can be split into several sessions: run some iterations, stop
(close the terminal, power off, ...), then continue where you left off. The
easiest way is `ruby bin/train --resume` (same flags as the original run),
which loads the latest adapter checkpoint before training; the manual
equivalent is adding `--resume-adapter-file _mlx/adapters_sft/adapters.safetensors`
to the `mlx_lm.lora` command above. Checkpoints are saved every 100 iters, so
a stopped run never loses more than a few minutes of work.

Prefer one long run over chained sessions whenever you can, for two reasons:

- **The optimizer state is not saved.** Resuming reloads only the adapter
  *weights*; Adam's momentum starts from zero again, so each resumed session
  begins with a small loss bump while the optimizer re-converges. At a low
  learning rate (1e-5) this is mild, but it still costs a bit of progress
  per session.
- **Data coverage.** Each session samples a random subset of the training
  set — no repeats within a session, but no memory across sessions either.
  With 20k examples, one run of 20k iterations sees **every example exactly
  once**; four chained runs of 5k iterations each see only ~63% of the
  examples at least once (the rest are never seen, some are seen twice). If
  covering the whole dataset matters, train in one go (`--iters 20000`).

Use chained sessions when you need to pause — e.g. to free the machine
overnight — and accept the coverage tradeoff, or resume a run that was
interrupted for any reason.

### Stage 3 — Use and evaluate the result

```bash
# Chat with the trained model (adapter loaded on top of the base):
.venv/bin/mlx_lm.generate \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path _mlx/adapters_sft \
  --max-tokens 300 \
  --prompt "Explain the difference between `include` and `prepend` in Ruby modules?"

# Bake the adapter into a standalone model (no base needed afterwards):
.venv/bin/mlx_lm.fuse \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path _mlx/adapters_sft \
  --save-path _mlx/model_sft
```

(This repo's one-command shortcut is `ruby bin/export` — fuses the latest
adapter into `_mlx/model_qwen3_export`, keeps the base's quantization, and
writes atomically; see the README.)

**Evaluation checklist** (before vs after, same prompts/seed):

1. A handful of Ruby coding questions (like the one above) — compare side by
   side with the base model.
2. General instruction-following questions — make sure the model didn't
   forget how to chat (important after CPT on an instruct model).
3. (Optional, rigorous) **lm-eval-harness** with the Ruby track:

   ```bash
   .venv/bin/pip install lm-eval
   .venv/bin/lm_eval --model mlx_lm \
     --model_args path=Qwen/Qwen2.5-3B-Instruct,adapter_path=_mlx/adapters_sft \
     --tasks humaneval_ruby \
     --num_fewshot 0
   ```

   (Check the current lm-eval docs for the exact MLX flags; run the same
   command without `adapter_path` for the baseline.)

### Stage 4 — Quantize the trained model (optional)

Bake the adapter into a standalone model, then produce a 4-bit version.
`mlx_lm.convert` ships with mlx-lm — no extra installs, runs
locally in minutes.

```bash
# 1. Fuse the adapter into a standalone model (~6 GB, fp16):
.venv/bin/mlx_lm.fuse \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path _mlx/adapters_sft \
  --save-path _mlx/model_sft

# 2. Quantize — 4-bit (~2 GB for the 3B):
.venv/bin/mlx_lm.convert \
  --mlx-path _mlx/model_sft \
  -q --q-bits 4 \
  --output-dir _mlx/model_sft_4bit

# 3. Use the quantized model (standalone, no adapter needed):
.venv/bin/mlx_lm.generate --model _mlx/model_sft_4bit --max-tokens 300 --prompt "..."
```

Notes:

- `--mlx-path` re-quantizes an already-converted MLX model (current mlx-lm);
  `--hf-path` also accepts a local directory if your version prefers it.
- 4-bit is the usual sweet spot for local inference.
  MLX runs quantized weights natively — no dequantization at inference.
- Fusing can be skipped: LoRA adapters apply on top of any base, including a
  quantized one — `mlx_lm.generate --model <base_4bit> --adapter-path
  _mlx/adapters_sft` works and keeps the adapter swappable. Fuse only when
  you want one self-contained file.

---

## 4. One-shot manual walkthrough (copy-paste, in order)

```bash
# ---- install ----
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install mlx-lm
.venv/bin/pip install "huggingface_hub[cli]"
.venv/bin/huggingface-cli download Qwen/Qwen2.5-3B-Instruct

# ---- verify ----
.venv/bin/python -c "import importlib.metadata as md; print(md.version('mlx'), md.version('mlx-lm'))"
.venv/bin/python -c "from mlx_lm import load; m, t = load('Qwen/Qwen2.5-3B-Instruct'); print('model loaded OK')"

# ---- data (Ruby) ----
ruby bin/build
ruby build/build_pretrain_corpus.rb
ruby build/build_sft_pairs.rb
ruby build/build_mlx_data.rb 12000 1000

# ---- stage 1: continued pretraining (optional) ----
.venv/bin/mlx_lm.lora --model Qwen/Qwen2.5-3B-Instruct --train \
  --data _pretrain/ruby_corpus.jsonl --adapter-path _mlx/adapters_cpt \
  --iters 1000 --batch-size 1 --learning-rate 1e-5 \
  --steps-per-report 50 --steps-per-eval 250 --max-seq-length 2048

# ---- stage 1 → 2 bridge (only if stage 1 ran) ----
.venv/bin/mlx_lm.fuse --model Qwen/Qwen3-8B-MLX-4bit \
  --adapter-path _mlx/adapters_cpt --save-path _mlx/model_cpt

# ---- stage 2: SFT ----
.venv/bin/mlx_lm.lora --model _mlx/model_cpt --train \
  --data _mlx --adapter-path _mlx/adapters_sft \
  --iters 1000 --batch-size 1 --learning-rate 1e-5 \
  --steps-per-report 10 --steps-per-eval 100 --max-seq-length 2048

# ---- stage 3: use ----
.venv/bin/mlx_lm.generate --model Qwen/Qwen3-8B-MLX-4bit \
  --adapter-path _mlx/adapters_sft --max-tokens 300 \
  --prompt "Explain the difference between `include` and `prepend` in Ruby modules?"

# ---- stage 4: quantize (optional) ----
.venv/bin/mlx_lm.fuse --model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path _mlx/adapters_sft --save-path _mlx/model_sft
.venv/bin/mlx_lm.convert --mlx-path _mlx/model_sft -q --q-bits 4 \
  --output-dir _mlx/model_sft_4bit
.venv/bin/mlx_lm.generate --model _mlx/model_sft_4bit --max-tokens 300 \
  --prompt "Explain the difference between `include` and `prepend` in Ruby modules?"
```

When it finishes: `_mlx/model_sft/` is a standalone, trained model you can
run with `mlx_lm.generate --model _mlx/model_sft`; `_mlx/model_sft_4bit/` is
its quantized version.

---

## 5. Reference

### File map

```
ruby-trainer/
├── TRAIN.md                  ← you are here
├── README.md                 ← pipeline + scripts + licensing
├── bin/build                 ← one-shot data pipeline (docs + code + datasets + SFT)
├── bin/train                 ← rebuild slice + run LoRA training
├── build/                    ← pipeline scripts (main.rb + build_*.rb + create_dataset.rb)
├── .venv/                    ← Python + MLX (created by you)
├── _sources/                 ← git clones of the source repositories
├── code_context/  docs_context/  _pretrain/  _dataset/
└── _mlx/                     ← MLX artifacts: train/valid jsonl, adapters, fused model
```

### Useful `mlx_lm.lora` flags

| Flag | Meaning |
|---|---|
| `--model` | Base model (HF id or local path) |
| `--train` | Run training (without it, the command evaluates only) |
| `--data` | JSONL file or directory with `train.jsonl`/`valid.jsonl` |
| `--adapter-path` | Where adapters are loaded from / saved to |
| `--iters` | Number of training iterations |
| `--batch-size` | 1 for small data; raise only if RAM allows |
| `--learning-rate` | 1e-5 is a safe default for this kind of data |
| `--steps-per-report` / `--steps-per-eval` | Logging / validation frequency |
| `--max-seq-length` | Truncation length — 2048 is enough with the pre-split data |
| `--save-every` | Adapter checkpoint frequency (default 100) |

### Troubleshooting

| Symptom | Fix |
|---|---|
| `mlx` has no `__version__` | Use `importlib.metadata.version('mlx')` for the version check |
| "Some sequences are longer than N tokens" | Shouldn't happen with the pre-split data — only if you set `--max-seq-length` below 2048 or use custom data |
| Pip refuses to install (externally-managed) | Use the venv (`python3 -m venv .venv`) — never install into system Python |
| Training is slow | Normal: ~150-180 tokens/sec on a 3B on Apple Silicon; drop `--steps-per-eval` frequency |
| Out of memory | Reduce `--max-seq-length`, `--batch-size`, or use a 4-bit quantization config for bigger models |
| Model re-downloads every run | It shouldn't — files live in `~/.cache/huggingface/hub/`; check free disk space |
| Wrong chat behavior after training | The chat template must match the model's — prefer `messages`-format data (as `build_mlx_data.rb` produces) |

### Switching to the 9B target model

The exact same commands work with `--model ornith-ai/Ornith-1.0-9B` (and
`huggingface-cli download ornith-ai/Ornith-1.0-9B`). Before the first run,
check its HuggingFace page for: base architecture (tool compatibility), chat
template, context length, and license. Expect ~2-3× the memory and time of
the 3B runs.

### Licensing

Training and running a model locally on this data is permitted by the source
licenses (see `README.md` → "Licensing"). Before **publishing or serving** the
trained model, review each source repository's license (ShareAlike and
non-compete clauses apply).
