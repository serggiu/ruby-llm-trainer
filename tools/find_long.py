#!/usr/bin/env python3
"""List entries longer than a token threshold (diagnostic helper)."""
import json
import sys
from transformers import AutoTokenizer

path, model_dir = sys.argv[1], sys.argv[2]
max_len = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 20

tok = AutoTokenizer.from_pretrained(model_dir)
shown = 0
for i, line in enumerate(open(path, encoding="utf-8"), 1):
    obj = json.loads(line)
    msgs = obj.get("messages")
    if msgs is None:
        msgs = [
            {"role": "user" if c["from"] == "human" else "assistant", "content": c["value"]}
            for c in obj["conversations"]
        ]
    text = tok.apply_chat_template(msgs, tokenize=False)
    n = len(tok.encode(text))
    if n > max_len:
        shown += 1
        chars = sum(len(m["content"]) for m in msgs)
        prompt = msgs[0]["content"][:120].replace("\n", " ")
        print(f"line {i}: {n} tokens, {chars} chars (c={chars/n:.2f}) | {prompt}")
        if shown >= limit:
            break
print(f"(showing first {shown})")
