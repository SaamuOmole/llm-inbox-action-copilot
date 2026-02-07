from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from inboxcopilot.llm.providers import OllamaProvider
from inboxcopilot.llm.validate_json import extract_json_object, validate_pred

PROMPT_PATH = Path("src/inboxcopilot/llm/prompts/intent_action_v1.txt")
GOLD_PATH = Path("data/gold/gold_labeled.jsonl")
PROC_PATH = Path("data/processed/emails_weak_labeled.jsonl")
OUT_PATH = Path("data/predictions/intent_action_v1.jsonl")


def load_index_jsonl(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    idx = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            idx[r[key]] = r
    return idx


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prompt_tmpl = PROMPT_PATH.read_text(encoding="utf-8")
    gold_idx = load_index_jsonl(GOLD_PATH, "email_id")
    proc_idx = load_index_jsonl(PROC_PATH, "email_id")

    provider = OllamaProvider(model="llama3.1:8b")  # change model name if needed

    n_ok = 0
    n_err = 0

    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for email_id, gold in gold_idx.items():
            if email_id not in proc_idx:
                # Should be rare; means processed dataset changed
                fout.write(json.dumps({
                    "email_id": email_id,
                    "error": "missing_in_processed"
                }) + "\n")
                n_err += 1
                continue

            rec = proc_idx[email_id]
            subject = rec.get("subject", "")
            body = (rec.get("body_clean") or "").strip()

            prompt = prompt_tmpl.format(subject=subject, body=body)

            try:
                raw = provider.generate(prompt)
                obj = extract_json_object(raw)
                pred = validate_pred(obj)

                out = {
                    "email_id": email_id,
                    "intent_pred": pred["intent"],
                    "action_present_pred": pred["action_present"],
                    "model": provider.model,
                    "prompt_version": "intent_action_v1",
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_ok += 1

            except Exception as e:
                fout.write(json.dumps({
                    "email_id": email_id,
                    "error": str(e)[:500],
                }, ensure_ascii=False) + "\n")
                n_err += 1

    print(f"Wrote predictions to {OUT_PATH}")
    print(f"OK: {n_ok} | ERR: {n_err}")


if __name__ == "__main__":
    main()
