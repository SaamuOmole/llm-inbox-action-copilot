from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st


INTENTS = [
    "needs_reply",
    "meeting_request",
    "invoice_payment",
    "action_required",
    "info_only",
    "newsletter",
]

INTENT_DEFS = {
    "meeting_request": (
        "Primary purpose is to schedule/reschedule/confirm a meeting/event. "
        "If the right next step is opening your calendar, choose this."
    ),
    "needs_reply": (
        "Requires a written response and replying completes the work. "
        "No separate task/calendar/payment action needed beyond replying."
    ),
    "action_required": (
        "Requires doing something beyond replying (e.g., review/approve/sign/submit). "
        "Reply may be optional or just acknowledgement; the work is a task."
    ),
    "invoice_payment": (
        "Involves paying money, invoices, receipts, billing, or payment due."
    ),
    "info_only": (
        "Informational only; no action expected (FYI, notifications without requests)."
    ),
    "newsletter": (
        "Bulk/marketing/automated email (often contains 'unsubscribe' and promotional content)."
    ),
}

ACTION_PRESENT_DEF = (
    "True if you need to do anything to progress the world (reply, schedule, pay, complete a task, follow up). "
    "False if it's purely informational/newsletter and you can ignore it without consequence."
)

DECISION_TREE = """\
Decision tree (pick first match):
1) Bulk/marketing? → newsletter
2) Scheduling the main outcome? → meeting_request
3) Money needs to be paid? → invoice_payment
4) Concrete task beyond replying? → action_required
5) Reply alone completes the work? → needs_reply
6) Otherwise → info_only

Rule of thumb:
- Unsure between needs_reply vs action_required?
  Ask: 'If I reply immediately, is my work done?'
  Yes → needs_reply, No → action_required
"""


SAMPLE_PATH = Path("data/gold/gold_sample_200.jsonl")
OUT_PATH = Path("data/gold/gold_labeled.jsonl")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _index_labeled(labeled_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # map email_id -> latest label (if you relabel, last one wins)
    idx = {}
    for r in labeled_rows:
        idx[r["email_id"]] = r
    return idx


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main():
    st.set_page_config(page_title="InboxCopilot — Gold Labeling", layout="wide")
    st.title("InboxCopilot — Gold Labeling (Intent + Action Presence)")

    if not SAMPLE_PATH.exists():
        st.error(f"Missing sample file: {SAMPLE_PATH}. Run sample_gold_set.py first.")
        st.stop()

    sample = _load_jsonl(SAMPLE_PATH)
    labeled = _load_jsonl(OUT_PATH)
    labeled_idx = _index_labeled(labeled)

    # Determine which items are still unlabeled
    unlabeled = [r for r in sample if r["email_id"] not in labeled_idx]
    done = len(sample) - len(unlabeled)

    st.sidebar.header("Progress")
    st.sidebar.metric("Total", len(sample))
    st.sidebar.metric("Done", done)
    st.sidebar.metric("Remaining", len(unlabeled))
    
    st.sidebar.divider()
    st.sidebar.header("Label guide")
            
    exp = st.sidebar.expander("Intent definitions", expanded=True)
    with exp:
        for k in INTENTS:
            st.markdown(f"**{k}** — {INTENT_DEFS[k]}")
        
    exp2 = st.sidebar.expander("action_present_gold meaning", expanded=False)
    with exp2:
        st.write(ACTION_PRESENT_DEF)
    
    exp3 = st.sidebar.expander("Decision tree", expanded=False)
    with exp3:
        st.code(DECISION_TREE)

    if len(unlabeled) == 0:
        st.success("All sampled emails are labeled. Nice.")
        st.stop()

    # Navigation
    if "cursor" not in st.session_state:
        st.session_state.cursor = 0

    # Keep cursor in range
    st.session_state.cursor = max(0, min(st.session_state.cursor, len(unlabeled) - 1))
    rec = unlabeled[st.session_state.cursor]

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Email")
        st.markdown(f"**Subject:** {rec.get('subject','')}")
        st.markdown(f"**From:** {rec.get('from','')}")
        st.markdown(f"**Date:** {rec.get('date_iso') or rec.get('date_raw') or ''}")
        st.divider()

        body = rec.get("body_clean") or ""
        st.text_area("body_clean", body, height=420)

        with st.expander("Show raw body (body_raw)"):
            st.text_area("body_raw", rec.get("body_raw") or "", height=250)

    with right:
        st.subheader("Labels")

        st.caption("Weak labels (auto)")
        st.code(
            json.dumps(
                {
                    "intent_weak": rec.get("intent_weak"),
                    "action_present_weak": rec.get("action_present_weak"),
                    "hints": rec.get("hints", {}),
                },
                indent=2,
            ),
            language="json",
        )

        st.divider()

        # Gold labels to fill
        default_intent = rec.get("intent_weak") if rec.get("intent_weak") in INTENTS else "info_only"
        # intent_gold = st.selectbox("intent_gold", INTENTS, index=INTENTS.index(default_intent))
        st.caption("Gold intent: choose the primary outcome of the email")
        intent_gold = st.selectbox(
            "intent_gold",
            INTENTS,
            index=INTENTS.index(default_intent),
            help=(
                "Use the sidebar definitions. "
                "Tip: if the next step is opening your calendar → meeting_request."
            ),
        )

        default_action = bool(rec.get("action_present_weak", False))
        # action_present_gold = st.checkbox("action_present_gold", value=default_action)

        st.caption("Gold action flag: do you need to do anything after reading?")
        action_present_gold = st.checkbox(
            "action_present_gold",
            value=default_action,
            help=ACTION_PRESENT_DEF,
        )
        # Show definition of the currently selected intent inline (super helpful)
        st.info(f"**{intent_gold}**: {INTENT_DEFS[intent_gold]}")

        notes = st.text_input("notes (optional)", value="")

        st.divider()

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("⬅️ Prev", use_container_width=True) and st.session_state.cursor > 0:
                st.session_state.cursor -= 1
                st.rerun()

        with c2:
            if st.button("Skip", use_container_width=True):
                st.session_state.cursor += 1
                st.rerun()

        with c3:
            if st.button("Save & Next", type="primary", use_container_width=True):
                out = {
                    "email_id": rec["email_id"],
                    "subject": rec.get("subject", ""),
                    "from": rec.get("from", ""),
                    "date_iso": rec.get("date_iso"),
                    "intent_gold": intent_gold,
                    "action_present_gold": action_present_gold,
                    "notes": notes,
                    "labeled_at": _now_iso(),
                    # Keep weak labels for later comparison/eval
                    "intent_weak": rec.get("intent_weak"),
                    "action_present_weak": rec.get("action_present_weak"),
                }
                _append_jsonl(OUT_PATH, out)
                st.session_state.cursor += 1
                st.rerun()


if __name__ == "__main__":
    main()
