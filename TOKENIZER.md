# TOKENIZER.md — verifying our tokenizer against the reference (Python)

`tools/tokenizer.rb` is a pure-Ruby re-implementation of the HuggingFace
byte-level BPE tokenizer. This file explains how to run the **original
Python tokenizer** and **our tokenizer** on the same text and compare the
outputs — a manual sanity check, no test framework needed.

The two tokenizers should print **identical** `count [token ids]` lines for
any text. (History: verified on 3,000 real pipeline entries — 0 count
mismatches, identical token ids on sampled entries.)

## Setup (one-time)

1. The Python environment with `transformers` — the `.venv` created for MLX
   (see [TRAIN.md](TRAIN.md)) already has it:
   ```bash
   .venv/bin/python -c "import importlib.metadata as md; print(md.version('transformers'))"
   ```
2. A model directory containing `tokenizer.json` — the Qwen3-8B MLX snapshot
   in the HuggingFace cache:
   ```bash
   MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-MLX-4bit/snapshots/$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-MLX-4bit/snapshots | head -1)
   ```
   (Any model with a `tokenizer.json` works; `tools/tokenizer.rb` is
   model-agnostic for the BPE part.)

## 1. Raw text

**Python (reference):**

```bash
.venv/bin/python -c 'import sys; from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained(sys.argv[1]); ids = t.encode(sys.argv[2]); print(len(ids), ids)' "$MODEL" 'Hello world'
```

**Ruby (ours):**

```bash
printf 'Hello world' | ruby tools/tokenizer.rb --ids "$MODEL"
```

Both print:

```
2 [9707, 1879]
```

## 2. Chat messages (chat template applied)

**Python (reference):**

```bash
.venv/bin/python -c 'import sys, json; from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained(sys.argv[1]); msgs = json.loads(sys.argv[2]); ids = t.encode(t.apply_chat_template(msgs, tokenize=False)); print(len(ids), ids)' "$MODEL" '[{"role": "user", "content": "Hello world"}, {"role": "assistant", "content": "Hi there!"}]'
```

**Ruby (ours):**

```bash
ruby tools/tokenizer.rb "$MODEL" --chat '[{"role": "user", "content": "Hello world"}, {"role": "assistant", "content": "Hi there!"}]' --ids
```

Both print:

```
19 [151644, 872, 198, 9707, 1879, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271, 13048, 1052, 0, 151645, 198]
```

Note the empty `<think>` block (`151667 … 151668`) around the final
assistant message — a Qwen3 template quirk both tokenizers reproduce.

## 3. Bulk check over a dataset (optional)

Compare per-entry counts for the first 200 entries of a dataset:

**Python:** counts to `/tmp/py_counts.txt`:

```bash
.venv/bin/python -c '
import json, sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(sys.argv[1])
n = 0
for line in open(sys.argv[2], encoding="utf-8"):
    obj = json.loads(line)
    raw = obj.get("conversations") or obj["messages"]
    msgs = [m if "role" in m else {"role": "user" if m["from"] == "human" else "assistant", "content": m["value"]} for m in raw]
    print(len(t.encode(t.apply_chat_template(msgs, tokenize=False))))
    n += 1
    if n == 200:
        break
' "$MODEL" _dataset/_full_ruby_dataset.jsonl > /tmp/py_counts.txt
```

**Ruby:** same entries through our tokenizer (as a library):

```bash
ruby -rjson -e '
require_relative "tools/tokenizer"
tok = Tokenizer.new(ARGV[1])
File.foreach(ARGV[0]).with_index do |line, i|
  break if i >= 200
  obj = JSON.parse(line)
  raw = obj["messages"] || obj["conversations"].map { |c| { "role" => c["from"] == "human" ? "user" : "assistant", "content" => c["value"] } }
  puts tok.count_messages(raw)
end' _dataset/_full_ruby_dataset.jsonl "$MODEL" > /tmp/rb_counts.txt
```

Compare:

```bash
diff /tmp/py_counts.txt /tmp/rb_counts.txt && echo "IDENTICAL: all 200 entries match"
```

## Known scope limits (expect differences only here)

- The Ruby chat template covers the standard message structure
  (system/user/assistant + generation prompt). Tool calls,
  `<tool_response>` blocks, and multi-step reasoning branches of exotic
  templates are **not** replicated — tokenizing such messages will differ.
- `"!"` genuinely maps to token id `0` in the Qwen3 vocab — an id of `0` in
  the output is correct, not a fallback.
