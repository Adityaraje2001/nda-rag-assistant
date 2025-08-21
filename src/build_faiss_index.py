# src/build_faiss_index.py
import os
import json
import numpy as np
import pandas as pd

# Ensure faiss-cpu is installed: pip install faiss-cpu
try:
    import faiss
except ImportError as e:
    raise RuntimeError("FAISS is not installed. Run: pip install faiss-cpu") from e

# -------- Config --------
DATA_DIR = "data"
EMB_PATH = os.path.join(DATA_DIR, "embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "metadata.csv")
CONFIG_PATH = os.path.join(DATA_DIR, "embed_config.json")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")

# Defaults if embed_config.json is missing
DEFAULTS = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "normalize_embeddings": True,   # you said normalize was true
}

def load_embed_config(config_path: str, defaults: dict):
    cfg = defaults.copy()
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        cfg.update(file_cfg)
        print(f"Loaded config from {config_path}: {cfg}")
    else:
        print(f"No {config_path} found. Using defaults: {cfg}")
    return cfg

def build_ip_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an inner-product FAISS index for already-normalized embeddings."""
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Built IndexFlatIP with dim={dim}. Added {index.ntotal} vectors.")
    return index

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) Load config (to confirm normalization + keep record)
    cfg = load_embed_config(CONFIG_PATH, DEFAULTS)
    normalized = bool(cfg.get("normalize_embeddings", True))
    if not normalized:
        print("Warning: Config indicates embeddings are NOT normalized, "
              "but you stated they are. Proceeding with IP index anyway.")
    else:
        print("Config indicates embeddings are normalized. Using IP index.")

    # 2) Load embeddings and metadata
    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(f"Embeddings file not found: {EMB_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    embeddings = np.load(EMB_PATH)
    meta = pd.read_csv(META_PATH)

    if embeddings.shape[0] != len(meta):
        raise ValueError(
            f"Row mismatch: embeddings={embeddings.shape} vs metadata={len(meta)}. "
            "They must be aligned row-by-row."
        )

    print(f"Loaded embeddings: {embeddings.shape}, metadata rows: {len(meta)}")

    # 3) Build FAISS index (inner product, since vectors are normalized)
    index = build_ip_index(embeddings)

    # 4) Save FAISS index
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved FAISS index to: {INDEX_PATH}")

    # 5) Quick sanity test: self-search first vector
    sample = embeddings[0:1].astype(np.float32)
    D, I = index.search(sample, k=5)
    print("Quick test (top-5 indices):", I.tolist())
    print("Scores:", D.tolist())
    print("Top-1 metadata preview:")
    print(meta.iloc[I[0]].to_dict())

if __name__ == "__main__":
    main()
