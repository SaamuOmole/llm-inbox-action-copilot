from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from inboxcopilot.ingest.parse_email import parse_eml_bytes
from inboxcopilot.ingest.clean_text import clean_email_body


def iter_eml_files(eml_dir: Path) -> Iterable[Path]:
    for p in eml_dir.rglob("*.eml"):
        yield p


def main():
    # eml_dir = Path("data/raw/eml")
    eml_dir = Path("/Users/samuel.omole/OneDrive - Science and Technology Facilities Council/inbox_download/Inbox_20260204-1501/Inbox")
    out_path = Path("data/processed/emails_clean.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for eml_file in iter_eml_files(eml_dir):
            raw_bytes = eml_file.read_bytes()
            rec = parse_eml_bytes(raw_bytes)

            body_clean = clean_email_body(rec.get("body_raw", ""))
            rec["body_clean"] = body_clean

            # fallback email_id for local exports
            if not rec["email_id"]:
                rec["email_id"] = f"local::{eml_file.as_posix()}"

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} emails to {out_path}")


if __name__ == "__main__":
    main()
