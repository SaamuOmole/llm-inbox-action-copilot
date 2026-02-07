from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main():
    path = Path("data/processed/emails_weak_labeled.jsonl")
    c_intent = Counter()
    c_action = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            c_intent[r.get("intent_weak", "unknown")] += 1
            c_action[r.get("action_present_weak", False)] += 1

    print("\nIntent (weak) distribution:")
    for k, v in c_intent.most_common():
        print(f"{k:15s} {v:6d}")

    print("\nAction present (weak) distribution:")
    for k, v in c_action.most_common():
        print(f"{str(k):5s} {v:6d}")


if __name__ == "__main__":
    main()