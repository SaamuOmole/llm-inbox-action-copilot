from __future__ import annotations

import json
from pathlib import Path

from inboxcopilot.labeling.weak_rules import weak_label_email


def main():
    inp = Path("data/processed/emails_clean.jsonl")
    out = Path("data/processed/emails_weak_labeled.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            labels = weak_label_email(rec.get("subject", ""), rec.get("body_clean", ""))
            rec.update(labels)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} labeled emails to {out}")


if __name__ == "__main__":
    main()
