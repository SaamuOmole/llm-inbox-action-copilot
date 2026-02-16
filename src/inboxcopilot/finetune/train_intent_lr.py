from __future__ import annotations

from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from inboxcopilot.finetune.dataset import load_gold_examples
from inboxcopilot.finetune.embed import get_or_build_embeddings, DEFAULT_EMBED_MODEL


OUT_DIR = Path("models/finetune")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    examples = load_gold_examples()
    texts = [e.text for e in examples]
    y = np.array([e.intent_gold for e in examples], dtype=object)

    X = get_or_build_embeddings("gold_texts", texts, model_name=DEFAULT_EMBED_MODEL)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []

    for train_idx, test_idx in cv.split(X, y):
        clf = LogisticRegression(max_iter=3000)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))

    print(f"[intent_lr] 5-fold CV Accuracy={np.mean(accs):.3f}")

    clf = LogisticRegression(max_iter=3000)
    clf.fit(X, y)

    out = OUT_DIR / "intent_lr.joblib"
    joblib.dump({"model": clf, "embed_model": DEFAULT_EMBED_MODEL}, out)
    print(f"[intent_lr] Saved {out}")


if __name__ == "__main__":
    main()
