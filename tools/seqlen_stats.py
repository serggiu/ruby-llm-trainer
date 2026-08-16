#!/usr/bin/env python3
"""Measure the token-length distribution of an MLX chat-format JSONL dataset.

Usage: seqlen_stats.py dataset.jsonl model_dir [max_len]

Reports sample count, min/mean/p50/p90/p99/max token lengths (chat-template
applied, exactly like mlx-lm does), and how many samples exceed max_len
(default 4096) — i.e. how much of the data gets truncated during training.
"""

import json
import sys
from transformers import AutoTokenizer


def main() -> None:
    path, model_dir = sys.argv[1], sys.argv[2]
    max_len = int(sys.argv[3]) if len(sys.argv) > 3 else 4096

    tok = AutoTokenizer.from_pretrained(model_dir)
    lens = []
    bad = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            msgs = obj.get("messages")
            if msgs is None:
                # ShareGPT format: {"conversations": [{from, value}, ...]}
                msgs = [
                    {"role": "user" if c["from"] == "human" else "assistant",
                     "content": c["value"]}
                    for c in obj["conversations"]
                ]
            text = tok.apply_chat_template(msgs, tokenize=False)
            lens.append(len(tok.encode(text)))
        except Exception as e:
            bad += 1
            print(f"[warn] skipping entry: {e}")

    lens.sort()
    n = len(lens)
    if n == 0:
        print("no samples")
        return

    def pct(p):
        return lens[min(n - 1, int(n * p))]

    over = sum(1 for l in lens if l > max_len)
    print(f"samples: {n} (skipped {bad})")
    print(f"tokens:  min {lens[0]}  mean {sum(lens) // n}  "
          f"p50 {pct(0.5)}  p90 {pct(0.9)}  p99 {pct(0.99)}  max {lens[-1]}")
    print(f"> {max_len} tokens (truncated during training): {over} "
          f"({over / n * 100:.1f}%)")
    if over:
        print("longest few: " + ", ".join(str(l) for l in lens[-5:]))


if __name__ == "__main__":
    main()
