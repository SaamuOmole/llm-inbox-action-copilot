from __future__ import annotations
import re

# REPLY_SEPARATORS = [
#     r"^On .* wrote:$",
#     r"^From:.*$",
#     r"^Sent:.*$",
#     r"^To:.*$",
#     r"^Subject:.*$",
#     r"^-{2,}\s*Original Message\s*-{2,}$",
# ]

# Common reply/forward boundary markers across clients.
RE_BOUNDARY = re.compile(
    r"(?im)^(?:"
    r"on .{0,200}wrote:\s*$|"                  # On Tue, ... wrote:
    r"from:\s.*$|sent:\s.*$|to:\s.*$|cc:\s.*$|subject:\s.*$|"  # Outlook headers
    r"-----\s*original message\s*-----\s*$|"  # -----Original Message-----
    r"-----\s*forwarded message\s*-----\s*$|" # -----Forwarded message-----
    r"begin forwarded message:\s*$|"          # Apple Mail
    r"__{5,}\s*$|"                            # _________ separators
    r"-{5,}\s*$|"                             # ----- separators
    r"\*{5,}\s*$"                             # ***** separators
    r")"
)

RE_QUOTED_LINE = re.compile(r"(?m)^\s*>")  # traditional quoted reply

SIGNATURE_MARKERS = [
    r"^--\s*$",              # common signature delimiter
    r"^Sent from my .*",     # mobile signatures
]

_HTML_LIKE = re.compile(r"(?is)^\s*<!doctype\s+html|<html\b|<body\b|<head\b|<style\b|<table\b|</html>")

def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")

# def strip_quoted_reply(text: str) -> str:
#     lines = text.splitlines()
#     cleaned = []
#     for line in lines:
#         if any(re.match(pat, line.strip(), flags=re.IGNORECASE) for pat in REPLY_SEPARATORS):
#             break
#         # also stop if we hit a typical quote line prefix
#         if line.strip().startswith(">"):
#             break
#         cleaned.append(line)
#     return "\n".join(cleaned).strip()

def strip_quoted_reply(text: str) -> str:
    """
    Return only the newest content above the first detected quote boundary.
    """
    if not text:
        return ""

    # If the email uses '>' quoting, cut at the first quoted line.
    m = RE_QUOTED_LINE.search(text)
    if m:
        text = text[: m.start()]

    # Cut at the first boundary marker like "On ... wrote:" or "From:" block
    m = RE_BOUNDARY.search(text)
    if m:
        text = text[: m.start()]

    return text.strip()

def limit_length(text: str, max_chars: int = 4000, max_lines: int = 120) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines])
    if len(text) > max_chars:
        text = text[:max_chars]
    return text.strip()

def strip_signature(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if any(re.match(pat, line.strip(), flags=re.IGNORECASE) for pat in SIGNATURE_MARKERS):
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def clean_email_body(body_raw: str) -> str:
    txt = body_raw.replace("\r\n", "\n").replace("\r", "\n")
    
    # Safety net: if body looks like HTML, convert it to text first
    if _HTML_LIKE.search(txt or ""):
        txt = _html_to_text(txt)
        
    txt = strip_quoted_reply(txt)
    txt = strip_signature(txt)
    txt = normalize_whitespace(txt)
    txt = limit_length(txt)
    return txt
