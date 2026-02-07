from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    random.seed(42)
    inp = Path("data/processed/emails_weak_labeled.jsonl")
    out = Path("data/gold/gold_sample_200.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    buckets = defaultdict(list)

    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            body = (r.get("body_clean") or "").strip()
            if len(body) < 40:  # skip very short
                continue
            buckets[r.get("intent_weak", "unknown")].append(r)

    # Target counts per intent (adjust if some buckets are small)
    targets = {
        "meeting_request": 35,
        "invoice_payment": 35,
        "needs_reply": 60,
        "action_required": 30,
        "info_only": 25,
        "newsletter": 15,
    }

    sample = []
    for intent, k in targets.items():
        pool = buckets.get(intent, [])
        if not pool:
            continue
        take = min(k, len(pool))
        sample.extend(random.sample(pool, take))

    # If we didn't reach ~200, top up from remaining pool
    if len(sample) < 200:
        remaining = []
        chosen_ids = {r["email_id"] for r in sample}
        for intent, pool in buckets.items():
            for r in pool:
                if r["email_id"] not in chosen_ids:
                    remaining.append(r)
        topup = min(200 - len(sample), len(remaining))
        sample.extend(random.sample(remaining, topup))

    with out.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} records to {out}")


if __name__ == "__main__":
    main()
