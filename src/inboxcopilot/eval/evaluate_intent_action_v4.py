from __future__ import annotations

import os
import json
from collections import Counter, defaultdict
from pathlib import Path

GOLD_PATH = Path("data/gold/gold_labeled.jsonl")
# PRED_PATH = Path("data/predictions/intent_action_v4.jsonl")
PRED_PATH = Path(os.getenv("PRED_PATH", "data/predictions/intent_action_v4.jsonl"))
print(PRED_PATH)

INTENTS = [
    "needs_reply",
    "meeting_request",
    "invoice_payment",
    "action_required",
    "info_only",
    "newsletter",
]


def load_idx(path: Path, key: str) -> dict:
    idx = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            idx[r[key]] = r
    return idx


def main():
    gold = load_idx(GOLD_PATH, "email_id")
    preds = load_idx(PRED_PATH, "email_id")

    n = 0
    n_intent_ok = 0
    n_action_ok = 0

    # action detection metrics
    tp = fp = tn = fn = 0

    # intent confusion
    cm = defaultdict(lambda: Counter())

    skipped = 0

    for email_id, g in gold.items():
        p = preds.get(email_id)
        if not p or "error" in p:
            skipped += 1
            continue

        n += 1
        ig = g["intent_gold"]
        ip = p["intent_pred"]
        ag = bool(g["action_present_gold"])
        ap = bool(p["action_present_pred"])

        cm[ig][ip] += 1

        if ig == ip:
            n_intent_ok += 1
        if ag == ap:
            n_action_ok += 1

        if ag and ap:
            tp += 1
        elif (not ag) and ap:
            fp += 1
        elif (not ag) and (not ap):
            tn += 1
        elif ag and (not ap):
            fn += 1

    print(f"\nEvaluated: {n} (skipped/errors: {skipped})")

    if n == 0:
        print("No valid predictions to evaluate.")
        return

    print(f"\nIntent accuracy: {n_intent_ok/n:.3f}")
    print(f"Action accuracy: {n_action_ok/n:.3f}")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("\nAction detection (True=action_present):")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

    print("\nIntent confusion matrix (rows=gold, cols=pred):")
    header = "gold\\pred".ljust(18) + "".join(c.ljust(18) for c in INTENTS)
    print(header)
    for r in INTENTS:
        row = r.ljust(18)
        for c in INTENTS:
            row += str(cm[r][c]).ljust(18)
        print(row)


if __name__ == "__main__":
    main()
