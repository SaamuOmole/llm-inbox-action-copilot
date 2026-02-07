from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

INTENTS = {
    "needs_reply",
    "meeting_request",
    "invoice_payment",
    "action_required",
    "info_only",
    "newsletter",
}

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from a string.
    This handles models that accidentally add extra text.
    """
    text = text.strip()
    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object found in output: {text[:200]}")
    obj = json.loads(m.group(0))
    return obj


def validate_pred(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "intent" not in obj or "action_present" not in obj:
        raise ValueError(f"Missing keys in prediction: {obj}")

    intent = obj["intent"]
    if intent not in INTENTS:
        raise ValueError(f"Invalid intent '{intent}'. Must be one of {sorted(INTENTS)}")

    ap = obj["action_present"]
    if not isinstance(ap, bool):
        # accept 0/1 or "true"/"false" if model gets sloppy
        if ap in (0, 1):
            ap = bool(ap)
        elif isinstance(ap, str) and ap.lower() in ("true", "false"):
            ap = ap.lower() == "true"
        else:
            raise ValueError(f"action_present must be boolean, got: {ap!r}")

    return {"intent": intent, "action_present": ap}
