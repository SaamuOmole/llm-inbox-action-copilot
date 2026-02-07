from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

PRED_PATH = Path("data/predictions/intent_action_v4.jsonl")

def main():
    c = Counter()
    c_route_action = Counter()
    with PRED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            conf = r.get("stage1_confidence")
            c[conf] += 1
            # routed to action iff action_present_pred True
            c_route_action[conf] += 1 if r.get("action_present_pred") else 0

    print("Stage1 confidence distribution:")
    for k, v in c.items():
        print(k, v)

    print("\nRouted-to-action counts by confidence:")
    for k, v in c_route_action.items():
        print(k, v)

if __name__ == "__main__":
    main()
