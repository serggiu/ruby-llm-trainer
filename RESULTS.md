# Training Run Results

One table per **machine + model** combination, one row per run. Notes go
below the table, never as a column. Same machine and model? Add a row. New
machine or model? Add a new table. This keeps every run comparable before
committing hours of compute.

## Apple M4 (10-core) · 32 GB · Qwen3-8B-MLX-4bit (LoRA r8)

| Dataset | Iters | Total time | Avg speed | Peak mem | Final val loss |
|---|---|---|---|---|---|
| full (20,000 / 700) | 5,000 | 5 h 52 m | ~99 tok/s | 14.8 GB | 1.249 (best 1.014 @ iter 2800) |

**Notes:**

- Batch size 1, lr 1e-5, max-seq-length 2048, eval every 200 iters,
  checkpoints every 100 iters (50 saved).
- Final train loss ~1.0 (last report 1.106); ~2.09M tokens trained across
  the run (~25 % of the dataset seen — random sampling, no repeats within
  a run).
- Val loss improved from 2.079 (untrained) to a 1.01–1.58 band; it rose
  mildly in the final third (peak 1.582 @ iter 3600) then wobbled — a hint
  of late-run overfitting; the val-best checkpoint is iter 2800 (1.014).
- Speed depends on power state: battery + Low Power Mode throttled the first
  ~2 h to ~73 tok/s, ~110 tok/s after (hence the ~99 tok/s overall).
- Adapter size: 39 MB (LoRA weights only; base model untouched).
- Toolchain: Python 3.12 (`.venv`), mlx 0.32.0 / mlx-lm 0.31.3, Ruby 4.0.5;
  model snapshot `383413e9…`.

### Base-vs-adapter comparison (after the run)

4 prompts (3 Ruby + 1 general), same seed 42, max-tokens 300, default chat
template.

- **Base model:** stalled in `<think>` mode on 3/4 prompts and rambled past
  the token budget on the 4th — never delivered a finished answer.
- **Adapter:** answered all 4 directly (empty `<think>` wrappers); 2/4 fully
  correct (Rails migration, haiku), 2/4 with a hallucinated detail
  (`include`/`prepend` mechanics, `yield_self` attributed to `Proc`).

**Verdict:** the behavior win is real — the adapter answers Ruby questions
in one pass instead of stalling in thinking mode. Knowledge depth is still
limited, consistent with ~25 % dataset coverage at 5,000 iters: the model
learned the answer *behavior* before the *knowledge*. Expect sharper answers
after a full-coverage run (20,000 iters / one epoch).

### Checkpoint 2800 (val-best) vs the final adapter

Same 4 prompts, same seed 42. The two models converged behaviorally —
migration and haiku answers were identical — but differed on the knowledge
questions:

- **`include` vs `prepend`:** 2800 answered correctly with textbook examples
  (class method wins with `include`, module wins with `prepend`); the final
  adapter got the conclusion right but described the mechanism wrong
  (“added to the class's singleton class”).
- **`yield_self`:** 2800 was correct (Kernel module, real chaining examples,
  minor blemish: invalid `.yield` comparison); the final adapter hallucinated
  “defined on `Proc` objects”.

**Verdict:** the val-best checkpoint (2800) answers Ruby questions more
accurately than the final adapter. The extra 2,200 iterations added
hallucinated detail rather than accuracy — the late-run overfitting visible
in the val curve, confirmed in real answers. **Exported model: checkpoint
2800.**

## Adding a run

- **Same machine and model** — add a row to the existing table and extend its
  Notes.
- **Different machine or model** — copy the table into a new section, retitle
  it with the new hardware/model, and write that run's Notes below it.
- Keep notes honest about anything unusual (power state, throttling, resume
  sessions, custom settings).
