from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from inboxcopilot.rag.retrieve import search_similar, get_thread_context

RAG_DIR = Path(os.getenv("RAG_OUT_DIR", "data/rag"))
META_PATH = RAG_DIR / "meta_rag_labeled.jsonl"


def _one_line(s: str, max_len: int = 140) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= max_len else s[:max_len] + "..."


def _load_meta_map() -> Dict[str, Dict[str, Any]]:
    mp: Dict[str, Dict[str, Any]] = {}
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            mp[r["email_id"]] = r
    return mp


def build_label_aware_context(
    email_id: str,
    subject: str,
    body: str,
    k_similar: int = 6,
    max_thread_items: int = 4,
    max_examples: int = 4,
) -> str:
    """
    Returns context with:
    - thread snippets (unlabeled)
    - retrieved labeled examples (gold preferred, weak fallback)
    """
    meta_map = _load_meta_map()
    blocks: List[str] = []

    # 1) Thread context (no labels, just helps disambiguate content)
    thread = get_thread_context(email_id, max_items=max_thread_items)
    if thread:
        lines = []
        for t in thread:
            lines.append(
                f"- {t.get('date_iso','')[:10]} | From: {_one_line(t.get('from',''), 60)} | "
                f"Subj: {_one_line(t.get('subject',''), 80)} | Snip: {_one_line(t.get('body_snip',''), 160)}"
            )
        blocks.append("[Thread context]\n" + "\n".join(lines))

    # 2) Similar retrieved examples WITH labels
    query = f"Subject: {subject}\n\n{body}"
    sims: List[Tuple[float, Dict[str, Any]]] = search_similar(query, k=k_similar)

    examples = []
    for score, m in sims:
        rid = m.get("email_id")
        if not rid or rid == email_id:
            continue

        mm = meta_map.get(rid, {})
        # prefer gold labels if present
        intent = mm.get("intent_gold") or mm.get("intent_weak")
        ap = mm.get("action_present_gold")
        if ap is None:
            ap = mm.get("action_present_weak")

        if intent is None or ap is None:
            continue  # skip unlabeled examples for this block

        examples.append((score, mm, intent, bool(ap)))

    # sort by similarity descending and keep top N
    examples.sort(key=lambda x: x[0], reverse=True)
    examples = examples[:max_examples]

    if examples:
        lines = []
        for score, mm, intent, ap in examples:
            label_src = "gold" if mm.get("intent_gold") is not None else "weak"
            lines.append(
                f"- score={score:.3f} | label_src={label_src} | action_present={ap} | intent={intent} | "
                f"From: {_one_line(mm.get('sender',''), 50)} | Subj: {_one_line(mm.get('subject',''), 90)}"
            )
        blocks.append("[Similar labeled examples]\n" + "\n".join(lines))

    return "\n\n".join(blocks).strip()