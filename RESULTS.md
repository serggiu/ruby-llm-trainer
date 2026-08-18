# Training Run Results

One table per **machine + model** combination, one row per run. Notes go
below the table, never as a column. Same machine and model? Add a row. New
machine or model? Add a new table. This keeps every run comparable before
committing hours of compute.

## Apple M4 (10-core) · 32 GB · Qwen3-8B-MLX-4bit (LoRA r8)

| Dataset | Iters | Total time | Avg speed | Peak mem | Final val loss |
|---|---|---|---|---|---|
| full (20,000 / 700) | 5,000 | 5 h 52 m | ~99 tok/s | 14.8 GB | 1.249 (best 1.014 @ iter 2800) |
| stage 2 SFT — 164 / 41 (auto-split) | 200 | 30 min | ~68 tok/s | 12.4 GB | 1.126 (baseline 1.464 @ iter 1) |
| stage 3 SFT — 85 / 21 (auto-split) | 100 | ~13 min | ~74 tok/s | 11.3 GB | 0.611 (baseline 0.906 @ iter 1) |
| stage 4 SFT — 2,183 / 546 (auto-split) | 2,183 | ~2 h 46 m | ~106 tok/s | 15.6 GB | 0.977 (best 0.889 @ iter 1400) |
| stage 5 SFT — 1,264 / 316 (auto-split) | 1,264 | ~1 h 35 m | ~106 tok/s | 10.4 GB | 1.070 (best 0.708 @ iter 1200) |
| refresh capsules — 86 / 21 (auto-split) | 100 | ~15 min | ~105 tok/s | 16.9 GB | 1.178 (baseline 1.308 @ iter 1) |
| stage 7 SFT — 308 / 77 (auto-split) | 308 | ~19 min | ~111 tok/s | 10.3 GB | 0.923 (baseline 1.489 @ iter 1) |
| stage 8 SFT — 210 / 53 (auto-split) | 210 | ~15 min | ~85 tok/s | 10.1 GB | 1.080 (best 0.900 @ iter 200) |
| stage 9 SFT — 247 / 62 (auto-split) | 247 | ~16 min | ~85 tok/s | 9.1 GB | 1.262 (best 1.133 @ iter 200) |
| refresh capsules — 263 / 66 (auto-split) | 263 | ~20 min | ~85 tok/s | 18.9 GB | 1.108 (best 0.725 @ iter 200) |
| stage 11 SFT — 2,816 / 704 (auto-split) | 2,816 | ~4.8 h | ~88 tok/s | 13.1 GB | 1.173 (best 0.824 @ iter 2600) |
| stage 12 SFT — 458 / 115 (auto-split) | 458 | ~28 min | ~85 tok/s | 10.4 GB | 1.084 (best 0.930 @ iter 400) |
| stage 13 SFT — 346 / 86 (auto-split) | 346 | ~25 min | ~85 tok/s | 11.1 GB | 0.890 (baseline 1.334 @ iter 1) |
| stage 14 SFT — 4,474 / 1,119 (auto-split) | 4,474 | ~5.7 h | ~105 tok/s | 12.3 GB | 0.787 (best 0.751 @ iter 800) |

~ 70 tok/s with Battery Saver on
~ 100 tok/s with Battery Saver off

**Notes (run 1, full slice):**

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

**Notes (run 2):**

- SFT set 205 entries; auto-split kicked in (smaller than the proper slice):
  164 train / 41 valid.
- One epoch (200 iters ≈ 1.2 passes); speed lower (~68 tok/s) because the
  short examples make per-step overhead dominate.
- Val loss improved 1.464 → 1.126 on the held-out entries — real
  generalization, not memorization.
- Chat evaluation (3 testing-API questions): 2/3 excellent (writes idiomatic
  tests with correct assertion-argument order; correct value-vs-identity
  assertion semantics), 1/3 good but hallucinated a non-existent
  before/after class-level API for once-per-class setup. Style transfer from
  the source is visible (the source's distinctive code conventions).

**Notes (run 3):**

- SFT set 106 entries; auto-split: 85 train / 21 valid; continued the model
  with `--resume`.
- 100 iters ≈ 1.2 epochs; ~74 tok/s; peak mem 11.3 GB.
- Val loss improved 0.906 → 0.611 on the held-out entries.
- Chat evaluation (3 time-travel API questions): code generation excellent
  (a working frozen-time test using the block form), but 2/3 core API facts
  wrong (freeze/travel distinction, `return` semantics) — the distilled
  capsule was curated afterwards with the missed facts.

**Notes (run 4):**

- SFT set 2,729 entries; auto-split: 2,183 train / 546 valid; continued the
  model with `--resume`.
- One full epoch (2,183 iters); ~977k tokens trained; 21 checkpoints;
  ~106 tok/s; peak mem 15.6 GB.
- Val loss improved 1.124 → 0.977 (best 0.889 @ iter 1400, outlier low 0.643
  @ 2000); mild late-run wobble, same signature as run 1.
- Chat evaluation (3 framework-API questions): 2/3 excellent — canonical
  module DSL pattern (class + instance methods) and correct string/symbol
  key semantics; 1/3 correct concept with one broken trailing example.
  Best stage result so far.

**Notes (run 5):**

- SFT set 1,580 entries; auto-split: 1,264 train / 316 valid; continued the
  model with `--resume`.
- One full epoch (1,264 iters); ~560k tokens trained; 13 checkpoints;
  ~106 tok/s; peak mem 10.4 GB — lowest of all runs.
- Val loss improved 0.855 → 0.708 @ iter 1200 (−0.147, the cleanest
  improvement yet); final train loss 0.568.
- Sharpest late-run drift of any session: final val 1.070 vs best 0.708
  (+0.362) — the val-best checkpoint is the one to export.
- Chat evaluation (3 bootstrap/framework questions, exported checkpoint):
  2/3 good — a correct component-definition example (inheritance +
  initializer DSL) and a correct mountable-scaffold answer; 1/3 conflated
  config-file initializers with the component ordering DSL (`before:`/`after:`).
  Same signature as earlier runs: concepts right, edge details blur.

**Notes (run 6 — knowledge refresh):**

- Trained on the distilled capsules of all previously used sources
  (107 entries; 86 train / 21 valid), continuing from the val-best
  checkpoint of run 5 — nothing lost, just re-anchored.
- 100 iters ≈ 1.2 epochs; ~105 tok/s; peak mem 16.9 GB — the longest
  capsule entries sit near the 2048-token cap, which pushes memory up
  versus the ordinary runs.
- Val loss improved 1.308 → 1.178 on the held-out capsule entries, with
  the final train loss at 0.466 (content absorbed).
- Chat evaluation across all prior sources (6 previously-missed facts):
  5/6 now answered correctly after the refresh; 1/6 still wrong
  (once-per-class test setup — the corresponding capsule lacked that
  fact, curated afterwards for the next refresh).

**Notes (run 7):**

- SFT set 385 entries; auto-split: 308 train / 77 valid; continued the
  model with `--resume`. Bug-fix pairs are now scoped to the module being
  trained (this run: 0).
- One full epoch (308 iters); ~125k tokens; ~111 tok/s; peak mem 10.3 GB;
  ~19 min.
- Largest improvement of any session: val 1.489 → 0.923 (−0.566); the
  final adapter equals the best — the first session with no late-run drift.
- Chat evaluation (3 job-queue questions): 2/3 excellent (exact retry/discard
  API with options; idiomatic enqueue-assertion example), 1/3 good with one
  minor fabrication (a nonexistent `job` class macro).

**Notes (run 8):**

- SFT set 263 entries; auto-split: 210 train / 53 valid; continued the
  model with `--resume`; bug-fix pairs scoped to the module (2).
- One full epoch (210 iters); ~71k tokens; peak mem 10.1 GB (lowest yet);
  ~15 min; speed dipped to ~85 tok/s (battery-saver window).
- Second-largest improvement: val 1.324 → 0.900 @ iter 200 (−0.424), then
  late-run drift to 1.080 (+0.180) — val-best checkpoint 200 exported.
- Chat evaluation (3 rich-text questions, exported checkpoint): 1/3
  excellent (model-side rich-text setup, idiomatic), 1/3 good (editor and
  attachments concepts, truncated), 1/3 partial (storage schema and
  sanitizer attribution blurred). Usage surface strong, internals blur.

**Notes (run 9):**

- SFT set 309 entries; auto-split: 247 train / 62 valid; continued the
  model with `--resume`; bug-fix pairs scoped to the module (0).
- One full epoch (247 iters); ~85k tokens; peak mem 9.1 GB (lowest of all
  runs); ~16 min; ~85 tok/s (battery-saver windows).
- Smallest improvement of the recent stages: val 1.364 → 1.133 @ iter 200
  (−0.231), then late drift to 1.262 (+0.129) — val-best checkpoint 200.
- The hardest domain so far (protocol/streaming surface, server-side code
  only); chat evaluation (3 realtime questions, val-best checkpoint):
  2/3 excellent (channel definition + stream helpers; connection
  identification pattern), 1/3 partial (`broadcast_to` semantics blurred).

**Notes (run 10 — knowledge refresh #2):**

- All 7 knowledge capsules (329 entries); auto-split: 263 train / 66 valid;
  resumed from the previous stage's val-best checkpoint.
- One epoch (263 iters); ~75k tokens; ~85 tok/s; peak mem 18.9 GB — the
  longest capsule entries sit near the 2048-token cap.
- Best val ever recorded: 0.725 @ iter 200 (−0.409 from the 1.134 baseline),
  then late drift to 1.108; the val-best checkpoint was kept as the next
  session's starting point.
- Spot-check after the refresh: curated distinctions re-anchored correctly
  (time-travel semantics). Two findings: a once-per-class test-setup fact
  remains wrong after five attempts — logged as a permanent known
  limitation (a strong model prior beats LoRA at this scale); and a
  block-return fact absent from the capsules regressed — capsules were
  curated afterwards to cover it. Knowledge not in a capsule is not
  protected by refreshes.

**Notes (run 11 — two modules in one stage):**

- SFT set 3,520 entries; auto-split: 2,816 train / 704 valid; continued the
  model with `--resume`; bug-fix pairs scoped to the materialized modules.
- One full epoch (2,816 iters); 1.48M tokens; ~4.8 h; peak mem 13.1 GB;
  speed cycled 71–111 tok/s with power state (time, not learning).
- Strongest training-set eval of any stage: val 1.193 → 0.824 @ iter 2600
  (−0.369, on a 704-entry held-out set), with the familiar signature:
  two mid-run wobbles, then late drift to 1.173 — val-best checkpoint 2600.
- Chat evaluation (3 controller/routing/view questions, val-best
  checkpoint): 3/3 excellent — textbook CRUD with strong parameters and
  before_action scoping, nested resources with the generated route list,
  and the render-vs-redirect distinction with a partial example. The
  strongest chat result of any stage.

**Notes (run 12):**

- SFT set 573 entries; auto-split: 458 train / 115 valid; continued the
  model with `--resume`; bug-fix pairs scoped to the module (3).
- One full epoch (458 iters); ~169k tokens; peak mem 10.4 GB; ~28 min;
  speed dipped to ~67 tok/s at the end (battery-saver window).
- Solid improvement: val 1.279 → 0.930 @ iter 400 (−0.349), then late
  drift to 1.084 (+0.154) — val-best checkpoint 400.
- Chat evaluation (3 file-attachment questions, val-best checkpoint):
  2/3 good (attach/purge macros and the variant API exact), 1/3 with a
  fabricated backend list (services beyond the core Disk/S3/GCS/Azure set).
  Usage surface strong, internals blur.

**Notes (run 13 — two modules in one stage):**

- SFT set 432 entries; auto-split: 346 train / 86 valid; continued the
  model with `--resume`; bug-fix pairs scoped (0).
- One full epoch (346 iters); ~127k tokens; ~25 min; peak mem 11.1 GB;
  ~85 tok/s (battery-saver windows).
- Third-largest improvement: val 1.334 → 0.890 (−0.444); NO late-run
  drift — final equals best, the second fully-stable session (the other
  was also a small, well-scoped module stage).
- Chat evaluation (3 mail questions, final adapter): 1/3 good (mailer
  class, views, delivery), 2/3 partial — preview API signatures garbled
  and the inbound-routing DSL not delivered. The model has also begun
  mimicking the distilled "## API" answer style of the capsules, at times
  inventing signatures in that format — a format side-effect to watch.

**Notes (run 14):**

- SFT set 5,593 entries (the largest of any stage); auto-split:
  4,474 train / 1,119 valid; continued with `--resume`; scoped bug pairs
  (15). Highest method-implementation share of any module (1,118 pairs).
- One full epoch (4,474 iters); 2.19M tokens; ~5.7 h; peak mem 12.3 GB;
  steady ~105 tok/s.
- Strongest generalization of any stage: val 1.024 → 0.751 @ iter 800
  (−0.273) on the largest held-out set (1,119 entries); the late drift was
  the mildest ever (+0.036) — best training-set eval and best final of all
  runs. Val-best checkpoint 800 kept as the next session's starting point.
- Chat evaluation (3 ORM questions, val-best checkpoint): 3/3 excellent —
  associations with dependent options, reversible migrations with
  index/foreign-key DSL, and find/find_by/where semantics. The most
  consistent chat result of any stage.
- Note: from this stage on, `bin/train --resume` (and `bin/refresh`)
  automatically continue from the last session's val-best checkpoint
  instead of the drifted final.

### Base vs trained — 10-question comparison (after refresh #2)

10 questions spanning every trained area (test frameworks, time
manipulation, framework internals, job queues, rich text, realtime), same
seed 42, max-tokens 180, asked of both the base model and the exported
trained model.

- **Base model: 0/10 delivered** — stalled in `<think>` mode on every
  question and never produced a final answer within the budget (one
  sketched its example then stopped).
- **Trained model: 10/10 answered directly** — 8 excellent-to-good
  (correct class structures, exact retry/discard API, include-vs-prepend
  mechanics, ActionText setup, connection identification), 2 partial:
  the Timecop block-return value claim (since curated into the capsule)
  and an `assert_enqueued_with` example missing its canonical block form.

**Verdict:** the behavioral win is absolute — the trained model answers
Ruby questions in one pass where the base never finishes. Knowledge is
strong on the usage surface across all trained areas; the two blur spots
are the same details-level class the capsule refresh cycle exists to fix.

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

### Export finding (2026-08-17): 4-bit fusion loses the adapter — resolved

`mlx_lm.fuse` without `--dequantize` merges the LoRA deltas and then
re-quantizes the weights back to 4-bit inline — the small deltas are lost in
that step, and the exported model behaves like the **base** model (chat-test
of the fused model: it reverted to `<think>`-mode rambling, while the same
checkpoint answers directly when loaded as an adapter).

**Resolution:** the correct export is two-step — `mlx_lm.fuse --dequantize`
(fp16, ~16 GB) then `mlx_lm.convert -q --q-bits 4` on the fused weights.
Quantizing after fusing preserves the adapter, giving a correct 4-bit
standalone model (~4.6 GB). `bin/export` now does exactly this by default;
`--dequantize` keeps the fp16 version. Also fixed: `bin/export --checkpoint`
was not wired into the run (it always fused the latest adapter) — now
staged, validated, and tested.

## Adding a run

- **Same machine and model** — add a row to the existing table and extend its
  Notes.
- **Different machine or model** — copy the table into a new section, retitle
  it with the new hardware/model, and write that run's Notes below it.
- Keep notes honest about anything unusual (power state, throttling, resume
  sessions, custom settings).
