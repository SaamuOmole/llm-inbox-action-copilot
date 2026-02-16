from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import joblib

from inboxcopilot.finetune.dataset import load_gold_examples, load_jsonl_index
from inboxcopilot.finetune.embed import get_or_build_embeddings, DEFAULT_EMBED_MODEL


EMAILS_PATH = Path("data/processed/emails_clean.jsonl")
GOLD_PATH = Path("data/gold/gold_labeled.jsonl")

OUT_PATH = Path("data/predictions/lr_oof_embed_logreg.jsonl")
N_SPLITS = 5


def main():
    # Load gold examples (text + labels)
    examples = load_gold_examples()
    email_ids = [e.email_id for e in examples]
    texts = [e.text for e in examples]

    y_action = np.array([1 if e.action_present_gold else 0 for e in examples], dtype=int)
    y_intent = np.array([e.intent_gold for e in examples], dtype=object)

    # Cached embeddings for the whole gold set
    X = get_or_build_embeddings("gold_texts", texts, model_name=DEFAULT_EMBED_MODEL)

    # Out-of-fold storage
    oof_action_prob = np.zeros(len(examples), dtype=float)
    oof_action_pred = np.zeros(len(examples), dtype=int)
    oof_intent_pred = np.array([""] * len(examples), dtype=object)

    # Stratify folds on intent (multi-class) for better balance
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_intent), start=1):
        X_train, X_test = X[train_idx], X[test_idx]

        # Train action model (binary)
        action_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        action_clf.fit(X_train, y_action[train_idx])

        prob = action_clf.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)  # we can tune this later

        oof_action_prob[test_idx] = prob
        oof_action_pred[test_idx] = pred

        # Train intent model (6-class)
        intent_clf = LogisticRegression(max_iter=3000)
        intent_clf.fit(X_train, y_intent[train_idx])
        oof_intent_pred[test_idx] = intent_clf.predict(X_test)

        print(f"[fold {fold}] trained on {len(train_idx)} / predicted {len(test_idx)}")

    # Write predictions JSONL in the same schema as your other pipelines
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i, email_id in enumerate(email_ids):
            f.write(json.dumps({
                "email_id": email_id,
                "intent_pred": str(oof_intent_pred[i]),
                "action_present_pred": bool(oof_action_pred[i]),
                "action_prob": float(oof_action_prob[i]),
                "model": f"LR({DEFAULT_EMBED_MODEL})",
                "pipeline_version": f"v7_lr_oof_{N_SPLITS}fold",
            }, ensure_ascii=False) + "\n")

    print(f"Wrote out-of-fold predictions: {OUT_PATH}")


if __name__ == "__main__":
    main()
