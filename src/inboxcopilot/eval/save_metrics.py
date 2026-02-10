from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path


INTENTS = [
    "needs_reply",
    "meeting_request",
    "invoice_payment",
    "action_required",
    "info_only",
    "newsletter",
]


@dataclass
class Metrics:
    version: str
    model: str
    n: int
    intent_accuracy: float
    action_accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int
    action_precision: float
    action_recall: float
    action_f1: float


def load_index(path: Path, key: str) -> dict:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r[key]] = r
    return out


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def main():
    gold_path = Path(os.getenv("GOLD_PATH", "data/gold/gold_labeled.jsonl"))
    pred_path = Path(os.getenv("PRED_PATH"))  # must be set
    out_dir = Path(os.getenv("OUT_DIR", "data/results"))
    version = os.getenv("VERSION", "v4")

    if not pred_path:
        raise SystemExit("Set PRED_PATH to the predictions jsonl you want to score.")

    out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_index(gold_path, "email_id")

    # predictions jsonl may contain error lines — skip them
    preds = {}
    model_name = None
    with Path(pred_path).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            preds[r["email_id"]] = r
            if model_name is None:
                model_name = r.get("model", "unknown")

    # Evaluate on intersection
    ids = [eid for eid in gold.keys() if eid in preds]
    n = len(ids)

    # Confusion matrix
    cm = defaultdict(lambda: Counter())

    intent_correct = 0
    action_correct = 0

    tp = fp = tn = fn = 0

    for eid in ids:
        g = gold[eid]
        p = preds[eid]

        ig = g["intent_gold"]
        ip = p["intent_pred"]
        ag = bool(g["action_present_gold"])
        ap = bool(p["action_present_pred"])

        cm[ig][ip] += 1

        if ig == ip:
            intent_correct += 1
        if ag == ap:
            action_correct += 1

        if ag and ap:
            tp += 1
        elif (not ag) and ap:
            fp += 1
        elif (not ag) and (not ap):
            tn += 1
        elif ag and (not ap):
            fn += 1

    intent_acc = safe_div(intent_correct, n)
    action_acc = safe_div(action_correct, n)

    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)

    metrics = Metrics(
        version=version,
        model=model_name or "unknown",
        n=n,
        intent_accuracy=round(intent_acc, 3),
        action_accuracy=round(action_acc, 3),
        tp=tp, fp=fp, tn=tn, fn=fn,
        action_precision=round(prec, 3),
        action_recall=round(rec, 3),
        action_f1=round(f1, 3),
    )

    # Write metrics JSON
    safe_model = (metrics.model.replace(":", "_").replace("/", "_").replace(" ", "_"))
    out_path = out_dir / f"metrics__{version}__{safe_model}.json"
    out_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    # Also write confusion matrix (optional, useful later)
    cm_path = out_dir / f"cm__{version}__{safe_model}.json"
    cm_obj = {r: {c: int(cm[r][c]) for c in INTENTS} for r in INTENTS}
    cm_path.write_text(json.dumps(cm_obj, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"Wrote: {cm_path}")


if __name__ == "__main__":
    main()
