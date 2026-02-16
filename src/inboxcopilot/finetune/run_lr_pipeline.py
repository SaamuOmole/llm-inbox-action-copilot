from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import joblib

from inboxcopilot.finetune.dataset import load_jsonl_index
from inboxcopilot.finetune.embed import embed_texts


EMAILS_PATH = Path("data/processed/emails_clean.jsonl")
GOLD_PATH = Path("data/gold/gold_labeled.jsonl")
OUT_PATH = Path("data/predictions/lr_embed_logreg.jsonl")

MODELS_DIR = Path("models/finetune")
ACTION_MODEL_PATH = MODELS_DIR / "action_lr.joblib"
INTENT_MODEL_PATH = MODELS_DIR / "intent_lr.joblib"

def as_text(val) -> str:
    """Normalize common email fields that may be str | list[str] | list[dict] | dict | None."""
    if val is None:
        return ""

    # simple string
    if isinstance(val, str):
        return val.strip()

    # dict (e.g. {"name": "...", "email": "..."})
    if isinstance(val, dict):
        name = (val.get("name") or "").strip()
        email = (val.get("email") or "").strip()
        if name and email:
            return f"{name} <{email}>"
        return name or email

    # list/tuple of strings or dicts
    if isinstance(val, (list, tuple)):
        parts = [as_text(x) for x in val]
        parts = [p for p in parts if p]  # drop empties
        return ", ".join(parts)

    # fallback (numbers, etc.)
    return str(val).strip()


def make_text(rec: Dict[str, Any], max_body_chars: int = 4000) -> str:
    subject = as_text(rec.get("subject"))
    sender = as_text(rec.get("from"))
    to = as_text(rec.get("to"))
    body = as_text(rec.get("body_clean"))
    if len(body) > max_body_chars:
        body = body[:max_body_chars] + " ..."
    return f"Subject: {subject}\nFrom: {sender}\nTo: {to}\n\n{body}"


def main():
    gold = load_jsonl_index(GOLD_PATH, "email_id")
    emails = load_jsonl_index(EMAILS_PATH, "email_id")

    action_pack = joblib.load(ACTION_MODEL_PATH)
    intent_pack = joblib.load(INTENT_MODEL_PATH)
    action_clf = action_pack["model"]
    intent_clf = intent_pack["model"]

    # sanity: both should use same embed model
    embed_model = action_pack["embed_model"]
    assert embed_model == intent_pack["embed_model"], "Embed model mismatch"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ids = list(gold.keys())
    texts = []
    for email_id in ids:
        rec = emails[email_id]
        texts.append(make_text(rec))

    X = embed_texts(texts, model_name=embed_model)

    # action prediction
    action_prob = action_clf.predict_proba(X)[:, 1]
    action_pred = (action_prob >= 0.5).astype(int)  # later we'll tune this threshold using gold

    # intent prediction
    intent_pred = intent_clf.predict(X)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for email_id, ap, ap_p, intent in zip(ids, action_pred, action_prob, intent_pred):
            f.write(json.dumps({
                "email_id": email_id,
                "intent_pred": str(intent),
                "action_present_pred": bool(ap),
                "action_prob": float(ap_p),
                "model": f"LR({embed_model})",
                "pipeline_version": "v7_lr_baseline",
            }, ensure_ascii=False) + "\n")

    print(f"Wrote predictions: {OUT_PATH}")


if __name__ == "__main__":
    main()
