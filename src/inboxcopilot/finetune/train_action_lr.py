from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

from inboxcopilot.finetune.dataset import load_gold_examples
from inboxcopilot.finetune.embed import get_or_build_embeddings, DEFAULT_EMBED_MODEL


OUT_DIR = Path("models/finetune")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    examples = load_gold_examples()
    texts = [e.text for e in examples]
    y = np.array([1 if e.action_present_gold else 0 for e in examples], dtype=int)

    X = get_or_build_embeddings("gold_texts", texts, model_name=DEFAULT_EMBED_MODEL)

    # Simple CV sanity check (not strictly required but nice to report)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    precs, recs, f1s = [], [], []

    for train_idx, test_idx in cv.split(X, y):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        p, r, f1, _ = precision_recall_fscore_support(y[test_idx], pred, average="binary", zero_division=0)
        precs.append(p); recs.append(r); f1s.append(f1)

    print(f"[action_lr] 5-fold CV Precision={np.mean(precs):.3f} Recall={np.mean(recs):.3f} F1={np.mean(f1s):.3f}")

    # Fit final model on all gold
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X, y)

    out = OUT_DIR / "action_lr.joblib"
    joblib.dump({"model": clf, "embed_model": DEFAULT_EMBED_MODEL}, out)
    print(f"[action_lr] Saved {out}")


if __name__ == "__main__":
    main()
