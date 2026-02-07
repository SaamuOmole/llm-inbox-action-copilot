from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main():
    path = Path("data/gold/gold_labeled.jsonl")
    if not path.exists():
        print("No gold labels yet.")
        return

    c_intent = Counter()
    c_action = Counter()
    c_intent_match = 0
    c_action_match = 0
    n = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            c_intent[r["intent_gold"]] += 1
            c_action[r["action_present_gold"]] += 1
            if r.get("intent_gold") == r.get("intent_weak"):
                c_intent_match += 1
            if bool(r.get("action_present_gold")) == bool(r.get("action_present_weak")):
                c_action_match += 1

    print(f"\nGold labels: {n}")
    print("\nIntent (gold) distribution:")
    for k, v in c_intent.most_common():
        print(f"{k:15s} {v:6d}")

    print("\nAction present (gold) distribution:")
    for k, v in c_action.most_common():
        print(f"{str(k):5s} {v:6d}")

    print("\nWeak vs Gold agreement:")
    print(f"intent agreement: {c_intent_match/n:.3f}")
    print(f"action agreement: {c_action_match/n:.3f}")


if __name__ == "__main__":
    main()
