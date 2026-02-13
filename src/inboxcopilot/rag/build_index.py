from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


IN_PATH = Path(os.getenv("RAG_INPUT", "data/processed/emails_clean.jsonl"))
OUT_DIR = Path(os.getenv("RAG_OUT_DIR", "data/rag"))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")

MAX_BODY_CHARS = int(os.getenv("RAG_MAX_BODY_CHARS", "4000"))


@dataclass
class Meta:
    email_id: str
    thread_id: str
    sender: str
    to: str
    subject: str
    date_iso: str


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

        texts.append(make_doc(r))
        metas.append(
            Meta(
                email_id=email_id,
                thread_id=(r.get("thread_id") or ""),
                sender=(r.get("from") or ""),
                to=(r.get("to") or ""),
                subject=(r.get("subject") or ""),
                date_iso=(r.get("date_iso") or ""),
            )
        )

    if not texts:
        raise SystemExit(f"No docs built from {IN_PATH}")

    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product (normalized vectors)
    index.add(emb)

    faiss.write_index(index, str(OUT_DIR / "emails.faiss"))
    with (OUT_DIR / "meta.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

    print(f"Built FAISS index: {OUT_DIR/'emails.faiss'}")
    print(f"Wrote metadata:    {OUT_DIR/'meta.jsonl'}")
    print(f"Docs indexed:      {len(metas)}")


if __name__ == "__main__":
    main()
