from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple


EMAILS_PATH = Path("data/processed/emails_clean.jsonl")
GOLD_PATH = Path("data/gold/gold_labeled.jsonl")


@dataclass
class LabeledExample:
    email_id: str
    text: str
    intent_gold: str
    action_present_gold: bool
    

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


def _make_text(rec: Dict[str, Any], max_body_chars: int = 4000) -> str:
    subject = as_text(rec.get("subject"))
    sender = as_text(rec.get("from"))
    to = as_text(rec.get("to"))
    body = as_text(rec.get("body_clean"))

    if len(body) > max_body_chars:
        body = body[:max_body_chars] + " ..."

    return f"Subject: {subject}\nFrom: {sender}\nTo: {to}\n\n{body}"


def load_jsonl_index(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            idx[r[key]] = r
    return idx


def load_gold_examples() -> List[LabeledExample]:
    emails = load_jsonl_index(EMAILS_PATH, "email_id")
    gold = load_jsonl_index(GOLD_PATH, "email_id")

    out: List[LabeledExample] = []
    missing = 0

    for email_id, g in gold.items():
        rec = emails.get(email_id)
        if not rec:
            missing += 1
            continue

        intent = g.get("intent_gold")
        ap = g.get("action_present_gold")

        if intent is None or ap is None:
            continue

        out.append(
            LabeledExample(
                email_id=email_id,
                text=_make_text(rec),
                intent_gold=str(intent),
                action_present_gold=bool(ap),
            )
        )

    if missing:
        print(f"[dataset] Warning: {missing} gold ids missing in emails_clean.jsonl")

    return out
