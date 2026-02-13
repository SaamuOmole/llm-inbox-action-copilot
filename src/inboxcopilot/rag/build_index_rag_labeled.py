from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


IN_PATH = Path(os.getenv("RAG_INPUT", "data/processed/emails_clean.jsonl"))
OUT_DIR = Path(os.getenv("RAG_OUT_DIR", "data/rag"))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")

GOLD_PATH = Path(os.getenv("GOLD_PATH", "data/gold/gold_labeled.jsonl"))
WEAK_PATH = Path(os.getenv("WEAK_PATH", "data/processed/emails_weak_labeled.jsonl"))  # optional

MAX_BODY_CHARS = int(os.getenv("RAG_MAX_BODY_CHARS", "4000"))


@dataclass
class Meta:
    email_id: str
    thread_id: str
    sender: str
    to: str
    subject: str
    date_iso: str

    # label-aware fields (optional)
    intent_gold: Optional[str] = None
    action_present_gold: Optional[bool] = None
    intent_weak: Optional[str] = None
    action_present_weak: Optional[bool] = None


def load_index_jsonl(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    idx: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            idx[r[key]] = r
    return idx


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


def make_doc(rec: Dict[str, Any]) -> str:
    subject = as_text(rec.get("subject"))
    sender = as_text(rec.get("from"))
    to = as_text(rec.get("to"))
    body = as_text(rec.get("body_clean"))

    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + " ..."

    # Stable template helps retrieval
    return f"Subject: {subject}\nFrom: {sender}\nTo: {to}\n\n{body}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold_idx = load_index_jsonl(GOLD_PATH, "email_id")
    weak_idx = load_index_jsonl(WEAK_PATH, "email_id")  # may be empty

    records: List[Dict[str, Any]] = []
    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    texts: List[str] = []
    metas: List[Meta] = []

    for r in records:
        email_id = r.get("email_id")
        if not email_id:
            continue

        g = gold_idx.get(email_id, {})
        w = weak_idx.get(email_id, {})

        meta = Meta(
            email_id=email_id,
            thread_id=(r.get("thread_id") or ""),
            sender=(r.get("from") or ""),
            to=(r.get("to") or ""),
            subject=(r.get("subject") or ""),
            date_iso=(r.get("date_iso") or ""),
            intent_gold=g.get("intent_gold"),
            action_present_gold=g.get("action_present_gold"),
            intent_weak=w.get("intent_weak"),
            action_present_weak=w.get("action_present_weak"),
        )

        texts.append(make_doc(r))
        metas.append(meta)

    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, str(OUT_DIR / "emails_rag_labeled.faiss"))
    with (OUT_DIR / "meta_rag_labeled.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

    print(f"Built FAISS index: {OUT_DIR/'emails_rag_labeled.faiss'}")
    print(f"Wrote metadata:    {OUT_DIR/'meta_rag_labeled.jsonl'}")
    print(f"Docs indexed:      {len(metas)}")
    print(f"Gold labels found: {sum(1 for m in metas if m.intent_gold is not None)}")


if __name__ == "__main__":
    main()