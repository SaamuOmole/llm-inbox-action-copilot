from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any

from inboxcopilot.llm.providers import OllamaProvider, OpenAIProvider
from inboxcopilot.llm.validate_json import extract_json_object
from inboxcopilot.rag.label_context import build_label_aware_context


PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
provider = OpenAIProvider() if PROVIDER == "openai" else OllamaProvider()

PROMPT_ACTION = Path("src/inboxcopilot/llm/prompts/action_present_conf_v4.txt")
PROMPT_IF_ACTION = Path("src/inboxcopilot/llm/prompts/intent_if_action_v3.txt")
PROMPT_IF_NO_ACTION = Path("src/inboxcopilot/llm/prompts/intent_if_no_action_v3.txt")

GOLD_PATH = Path("data/gold/gold_labeled.jsonl")
PROC_PATH = Path("data/processed/emails_clean.jsonl")
OUT_PATH = Path(os.getenv("PRED_PATH", "data/predictions/intent_action_v6_labelrag.jsonl"))

ACTION_INTENTS = {"needs_reply", "meeting_request", "invoice_payment", "action_required"}
NO_ACTION_INTENTS = {"info_only", "newsletter"}
CONF_LEVELS = {"high", "medium", "low"}


def load_index_jsonl(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    idx = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            idx[r[key]] = r
    return idx


def parse_stage1(raw: str) -> tuple[bool, str]:
    obj = extract_json_object(raw)
    if "action_present" not in obj or "confidence" not in obj:
        raise ValueError(f"Stage1 missing keys: {obj}")

    ap = obj["action_present"]
    if isinstance(ap, bool):
        pass
    elif ap in (0, 1):
        ap = bool(ap)
    elif isinstance(ap, str) and ap.lower() in ("true", "false"):
        ap = ap.lower() == "true"
    else:
        raise ValueError(f"Stage1 action_present not boolean: {ap!r}")

    conf = obj["confidence"]
    if isinstance(conf, str):
        conf = conf.lower().strip()
    if conf not in CONF_LEVELS:
        raise ValueError(f"Stage1 confidence invalid: {conf!r}")

    return ap, conf


def parse_intent(raw: str, allowed: set[str]) -> str:
    obj = extract_json_object(raw)
    if "intent" not in obj:
        raise ValueError(f"Missing intent key: {obj}")
    intent = obj["intent"]
    if intent not in allowed:
        raise ValueError(f"Intent '{intent}' not in allowed set {sorted(allowed)}")
    return intent


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tmpl_action = PROMPT_ACTION.read_text(encoding="utf-8")
    tmpl_if_action = PROMPT_IF_ACTION.read_text(encoding="utf-8")
    tmpl_if_no_action = PROMPT_IF_NO_ACTION.read_text(encoding="utf-8")

    gold_idx = load_index_jsonl(GOLD_PATH, "email_id")
    proc_idx = load_index_jsonl(PROC_PATH, "email_id")

    n_ok = 0
    n_err = 0
    rag_used = 0

    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for email_id in gold_idx.keys():
            rec = proc_idx.get(email_id)
            if not rec:
                fout.write(json.dumps({"email_id": email_id, "error": "missing_in_processed"}) + "\n")
                n_err += 1
                continue

            subject = rec.get("subject", "")
            body = (rec.get("body_clean") or "").strip()

            try:
                # Stage 1 (NO RAG)
                p1 = tmpl_action.format(subject=subject, body=body, context="")
                raw1 = provider.generate(p1)
                ap, conf = parse_stage1(raw1)

                # Confidence-aware routing (same as v4)
                route_action = (ap is True) and (conf in ("high", "medium"))

                # Label-aware RAG context for Stage 2
                # Only use when not-high confidence OR short email
                use_rag = (conf in ("low", "medium")) or (len(body) < 400)
                context = ""
                if use_rag:
                    context = build_label_aware_context(
                        email_id=email_id,
                        subject=subject,
                        body=body,
                        k_similar=6,
                        max_examples=4,
                    )
                    rag_used += 1

                if route_action:
                    p2 = tmpl_if_action.format(subject=subject, body=body, context=context)
                    raw2 = provider.generate(p2)
                    intent = parse_intent(raw2, ACTION_INTENTS)
                else:
                    ap = False
                    p2 = tmpl_if_no_action.format(subject=subject, body=body, context=context)
                    raw2 = provider.generate(p2)
                    intent = parse_intent(raw2, NO_ACTION_INTENTS)

                out = {
                    "email_id": email_id,
                    "intent_pred": intent,
                    "action_present_pred": ap,
                    "stage1_confidence": conf,
                    "rag_used": use_rag,
                    "model": provider.model,
                    "pipeline_version": "v6_labelrag",
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_ok += 1

            except Exception as e:
                fout.write(json.dumps({"email_id": email_id, "error": str(e)[:800]}, ensure_ascii=False) + "\n")
                n_err += 1

    print(f"Wrote predictions to {OUT_PATH}")
    print(f"OK: {n_ok} | ERR: {n_err} | label-RAG used: {rag_used}/{n_ok+n_err}")


if __name__ == "__main__":
    main()
