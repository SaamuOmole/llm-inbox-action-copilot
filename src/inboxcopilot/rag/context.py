from __future__ import annotations

from typing import Any, Dict, List, Tuple

from inboxcopilot.rag.retrieve import search_similar, get_thread_context


def _one_line(s: str, max_len: int = 180) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def build_rag_context(
    email_id: str,
    subject: str,
    body: str,
    k_similar: int = 4,
    max_thread_items: int = 4,
) -> str:
    """
    Returns a compact text block you can inject into prompts as {context}.
    """
    blocks: List[str] = []

    # 1) Thread context
    thread = get_thread_context(email_id, max_items=max_thread_items)
    if thread:
        lines = []
        for t in thread:
            lines.append(
                f"- {t.get('date_iso','')[:10]} | From: {_one_line(t.get('from',''), 60)} | "
                f"Subj: {_one_line(t.get('subject',''), 80)} | Snip: {_one_line(t.get('body_snip',''), 160)}"
            )
        blocks.append("[Thread context]\n" + "\n".join(lines))

    # 2) Similar emails (semantic)
    query = f"Subject: {subject}\n\n{body}"
    sims: List[Tuple[float, Dict[str, Any]]] = search_similar(query, k=k_similar)

    if sims:
        lines = []
        for score, m in sims:
            # avoid leaking the current email back if it shows up
            if m.get("email_id") == email_id:
                continue
            lines.append(
                f"- score={score:.3f} | From: {_one_line(m.get('sender',''), 60)} | "
                f"Subj: {_one_line(m.get('subject',''), 90)}"
            )
        if lines:
            blocks.append("[Similar past emails]\n" + "\n".join(lines))

    if not blocks:
        return ""

    return "\n\n".join(blocks)