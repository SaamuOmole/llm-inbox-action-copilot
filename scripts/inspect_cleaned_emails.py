import json
import random

path = "data/processed/emails_clean.jsonl"

with open(path, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

sample = random.sample(rows, 5)

for i, r in enumerate(sample, 1):
    print("=" * 80)
    print(f"[{i}] SUBJECT: {r['subject']}")
    print("-" * 80)
    print(r["body_clean"])