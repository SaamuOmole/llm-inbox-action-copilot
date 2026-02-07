from __future__ import annotations
import json
from pathlib import Path

GOLD = Path("data/gold/gold_labeled.jsonl")

def main():
    rows = []
    with GOLD.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)

    print("\nINTENT DISAGREEMENTS\n" + "=" * 80)
    for r in rows:
        if r["intent_gold"] != r.get("intent_weak"):
            print(f"\nEMAIL ID: {r['email_id']}")
            print(f"weak: {r.get('intent_weak')} → gold: {r['intent_gold']}")
            print(f"subject: {r.get('subject','')}")
            print("-" * 80)

    print("\nACTION FLAG DISAGREEMENTS\n" + "=" * 80)
    for r in rows:
        if bool(r["action_present_gold"]) != bool(r.get("action_present_weak")):
            print(f"\nEMAIL ID: {r['email_id']}")
            print(f"weak: {r.get('action_present_weak')} → gold: {r['action_present_gold']}")
            print(f"subject: {r.get('subject','')}")
            print("-" * 80)

if __name__ == "__main__":
    main()
