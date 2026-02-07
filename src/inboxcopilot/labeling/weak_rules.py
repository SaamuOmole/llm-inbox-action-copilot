from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, Tuple


INTENTS = [
    "needs_reply",
    "meeting_request",
    "invoice_payment",
    "action_required",
    "info_only",
    "newsletter",
]


# Simple patterns. We'll iterate once you see results.
RE_NEWSLETTER = re.compile(r"\bunsubscribe\b|\bview in browser\b|\bnewsletter\b", re.IGNORECASE)
RE_MEETING = re.compile(r"\bmeet\b|\bmeeting\b|\bzoom\b|\bteams\b|\bcalendar\b|\bavailability\b", re.IGNORECASE)
RE_INVOICE = re.compile(r"\binvoice\b|\bpayment\b|\bdue\b|\bpayable\b|\bremittance\b", re.IGNORECASE)

# Basic "question / request" heuristics
RE_QUESTION = re.compile(r"\?\s*$")
RE_REQUEST_VERBS = re.compile(r"\bcan you\b|\bcould you\b|\bplease\b|\bwould you\b", re.IGNORECASE)

# Crude date/time and money hints (for later)
RE_TIME_HINT = re.compile(r"\b(\d{1,2}(:\d{2})?\s?(am|pm))\b", re.IGNORECASE)
RE_DATE_HINT = re.compile(r"\b(mon|tue|wed|thu|fri|sat|sun)\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE)
RE_MONEY_HINT = re.compile(r"[$£€]\s?\d+([,.\s]\d+)*|\b\d+(\.\d{2})?\s?(usd|gbp|eur)\b", re.IGNORECASE)


def weak_label_email(subject: str, body_clean: str) -> Dict[str, Any]:
    text = f"{subject}\n\n{body_clean}".strip()

    # 1) Newsletter
    if RE_NEWSLETTER.search(text):
        return {
            "intent_weak": "newsletter",
            "action_present_weak": False,
            "hints": _hints(text),
        }

    # 2) Invoice/payment
    if RE_INVOICE.search(text) and RE_MONEY_HINT.search(text):
        return {
            "intent_weak": "invoice_payment",
            "action_present_weak": True,
            "hints": _hints(text),
        }

    # 3) Meeting request
    if RE_MEETING.search(text) and (RE_TIME_HINT.search(text) or RE_DATE_HINT.search(text)):
        return {
            "intent_weak": "meeting_request",
            "action_present_weak": True,
            "hints": _hints(text),
        }

    # 4) Needs reply (question/request)
    looks_like_question = "?" in text or RE_REQUEST_VERBS.search(text) is not None
    if looks_like_question:
        return {
            "intent_weak": "needs_reply",
            "action_present_weak": True,
            "hints": _hints(text),
        }

    # 5) Action required catch-all (e.g., "please review", "please sign")
    if re.search(r"\breview\b|\bsign\b|\bapprove\b|\baction required\b", text, flags=re.IGNORECASE):
        return {
            "intent_weak": "action_required",
            "action_present_weak": True,
            "hints": _hints(text),
        }

    # 6) Default info-only
    return {
        "intent_weak": "info_only",
        "action_present_weak": False,
        "hints": _hints(text),
    }


def _hints(text: str) -> Dict[str, Any]:
    return {
        "time_hints": list({m.group(0) for m in RE_TIME_HINT.finditer(text)})[:5],
        "date_hints": list({m.group(0) for m in RE_DATE_HINT.finditer(text)})[:5],
        "money_hints": list({m.group(0) for m in RE_MONEY_HINT.finditer(text)})[:5],
    }