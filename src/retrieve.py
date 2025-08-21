# src/retrieve.py
import os
import json
import numpy as np
import pandas as pd

import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "metadata.csv")
CONFIG_PATH = os.path.join(DATA_DIR, "embed_config.json")

DEFAULTS = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "normalize_embeddings": True,
}

def load_embed_config(path: str, defaults: dict):
    cfg = defaults.copy()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg

class Retriever:
    def __init__(self, k: int = 5):
        self.cfg = load_embed_config(CONFIG_PATH, DEFAULTS)
        self.model = SentenceTransformer(self.cfg["model_name"])
        self.normalize = bool(self.cfg.get("normalize_embeddings", True))

        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f"Metadata not found at {META_PATH}")

        self.index = faiss.read_index(INDEX_PATH)
        self.meta = pd.read_csv(META_PATH)
        self.k = k

    def search(self, query: str, k: int = None):
        if not query or not query.strip():
            return []
        k = k or self.k
        q_emb = self.model.encode([query], normalize_embeddings=self.normalize)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        D, I = self.index.search(q_emb, k)
        results = []
        I_row = I[0].ravel()
        D_row = D.ravel()
        for rank, idx in enumerate(I_row):
            idx_int = int(np.asarray(idx).item())
            if idx_int < 0 or idx_int >= len(self.meta):
                continue
            row = self.meta.iloc[idx_int]
            results.append({
                "rank": rank + 1,
                "score": float(D_row[rank]),
                "contract_id": row.get("contract_id"),
                "chunk_id": row.get("chunk_id"),
                "text": row.get("text"),
            })
        return results
