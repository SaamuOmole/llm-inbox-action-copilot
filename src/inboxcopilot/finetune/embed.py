from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import joblib


DEFAULT_EMBED_MODEL = os.getenv("FT_EMBED_MODEL", "all-MiniLM-L6-v2")
CACHE_DIR = Path(os.getenv("FT_CACHE_DIR", "models/finetune"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def embed_texts(texts: List[str], model_name: str = DEFAULT_EMBED_MODEL) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype="float32")


def cache_path(tag: str, model_name: str) -> Path:
    safe = model_name.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{tag}__{safe}.joblib"


def get_or_build_embeddings(tag: str, texts: List[str], model_name: str = DEFAULT_EMBED_MODEL) -> np.ndarray:
    p = cache_path(tag, model_name)
    if p.exists():
        return joblib.load(p)
    X = embed_texts(texts, model_name=model_name)
    joblib.dump(X, p)
    return X
