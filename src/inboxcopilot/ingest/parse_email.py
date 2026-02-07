from __future__ import annotations

import re
from dataclasses import dataclass
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dateutil import parser as dtparser

import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


_HTML_LIKE = re.compile(r"(?is)^\s*<!doctype\s+html|<html\b|<body\b|<head\b|<style\b|<table\b|</html>")
def looks_like_html(s: str) -> bool:
    if not s:
        return False
    return _HTML_LIKE.search(s) is not None


def _decode_part(part) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def extract_text_from_email(msg: EmailMessage) -> Dict[str, str]:
    """
    Returns:
      - text: best-effort plain text
      - html: html (if present)
    """
    text_parts = []
    html_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp.lower():
                continue
            if ctype == "text/plain":
                text_parts.append(_decode_part(part))
            elif ctype == "text/html":
                html_parts.append(_decode_part(part))
    else:
        ctype = msg.get_content_type()
        if ctype == "text/plain":
            text_parts.append(_decode_part(msg))
        elif ctype == "text/html":
            html_parts.append(_decode_part(msg))

    text = "\n".join(t for t in text_parts if t.strip())
    html = "\n".join(h for h in html_parts if h.strip())
    return {"text": text, "html": html}


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def parse_eml_bytes(raw_bytes: bytes) -> Dict[str, Any]:
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)

    # payloads = extract_text_from_email(msg)
    # body_text = payloads["text"].strip()
    # if not body_text and payloads["html"].strip():
    #     body_text = html_to_text(payloads["html"]).strip()
        
    payloads = extract_text_from_email(msg)

    text_candidate = (payloads["text"] or "").strip()
    html_candidate = (payloads["html"] or "").strip()

    # Prefer plain text unless it looks like HTML.
    if text_candidate and not looks_like_html(text_candidate):
        body_text = text_candidate
    elif html_candidate:
        body_text = html_to_text(html_candidate).strip()
    else:
        # Last resort: strip tags from the "text" candidate if it looks like HTML
        body_text = html_to_text(text_candidate).strip() if looks_like_html(text_candidate) else text_candidate


    # Parse date robustly (keep original too)
    date_raw = str(msg.get("Date") or "").strip()
    date_iso = None
    if date_raw:
        try:
            date_iso = dtparser.parse(date_raw).isoformat()
        except Exception:
            date_iso = None

    to_field = str(msg.get("To") or "").strip()
    to_list = [t.strip() for t in re.split(r",|;", to_field) if t.strip()]

    return {
        "email_id": str(msg.get("Message-ID") or "").strip() or None,
        "thread_id": None,  # will add later with Gmail
        "subject": str(msg.get("Subject") or "").strip(),
        "from": str(msg.get("From") or "").strip(),
        "to": to_list,
        "date_raw": date_raw,
        "date_iso": date_iso,
        "body_raw": body_text,
    }