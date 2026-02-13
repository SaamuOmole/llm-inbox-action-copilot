from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


RAG_DIR = Path(os.getenv("RAG_OUT_DIR", "data/rag"))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")

INDEX_PATH = RAG_DIR / "emails.faiss"
META_PATH = RAG_DIR / "meta.jsonl"

# Your processed emails (for pulling snippets by id)
EMAILS_PATH = Path(os.getenv("RAG_INPUT", "data/processed/emails_clean.jsonl"))


def load_meta() -> List[Dict[str, Any]]:
    meta = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


def load_email_index() -> Dict[str, Dict[str, Any]]:
    idx = {}
    with EMAILS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            idx[r["email_id"]] = r
    return idx


def search_similar(query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
    index = faiss.read_index(str(INDEX_PATH))
    meta = load_meta()

    model = SentenceTransformer(EMBED_MODEL)
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    scores, idxs = index.search(q, k)
    out = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        out.append((float(score), meta[int(i)]))
    return out


def get_thread_context(email_id: str, max_items: int = 5) -> List[Dict[str, Any]]:
    emails = load_email_index()
    rec = emails.get(email_id)
    if not rec:
        return []

    thread_id = rec.get("thread_id") or ""
    if not thread_id:
        return []

    # Return other emails in the same thread (excluding current)
    thread = [
        r for r in emails.values()
        if (r.get("thread_id") == thread_id and r.get("email_id") != email_id)
    ]

    # Sort by date_iso if possible
    thread.sort(key=lambda r: (r.get("date_iso") or ""))

    # Keep most recent
    thread = thread[-max_items:]

    # Compact view
    compact = []
    for r in thread:
        compact.append({
            "email_id": r["email_id"],
            "date_iso": r.get("date_iso", ""),
            "from": r.get("from", ""),
            "subject": r.get("subject", ""),
            "body_snip": (r.get("body_clean") or "")[:300].replace("\n", " ").strip(),
        })
    return compact


if __name__ == "__main__":
    # quick test
    q = os.getenv("Q", "invoice payment due")
    res = search_similar(q, k=5)
    for score, m in res:
        print(f"{score:.3f}  {m['email_id']}  {m.get('sender','')}  {m.get('subject','')}")
