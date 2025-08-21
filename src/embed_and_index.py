# src/embed_and_index.py
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Config
INPUT_JSONL = os.path.join("preprocessed_chunks.jsonl")  # adjust if needed
OUTPUT_EMB = os.path.join("data", "embeddings.npy")
OUTPUT_META = os.path.join("data", "metadata.csv")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
NORMALIZE = True  # set False if you prefer raw embeddings

def load_chunks(jsonl_path):
    texts = []
    contract_ids = []
    chunk_ids = []
    raw_texts = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("contract_id")
            # contract_id might be a tuple if created via os.path.splitext earlier; ensure string
            if isinstance(cid, (list, tuple)):
                cid = cid[0]
            chunk_id = obj.get("chunk_id")
            text = obj.get("text", "")

            raw_texts.append(text)
            texts.append(text)
            contract_ids.append(str(cid))
            chunk_ids.append(str(chunk_id))

    return texts, contract_ids, chunk_ids, raw_texts

def batched_encode(model, texts, batch_size=64, normalize=True):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        batch_emb = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize
        )
        embs.append(batch_emb)
    return np.vstack(embs)

def main():
    # Create data dir if missing
    os.makedirs("data", exist_ok=True)

    # 1) Load chunks
    texts, contract_ids, chunk_ids, raw_texts = load_chunks(INPUT_JSONL)
    if len(texts) == 0:
        print(f"No records found in {INPUT_JSONL}. Exiting.")
        return
    print(f"Loaded {len(texts)} chunks.")

    # 2) Load embedding model
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 3) Compute embeddings (batched)
    embeddings = batched_encode(model, texts, batch_size=BATCH_SIZE, normalize=NORMALIZE)
    print(f"Embeddings shape: {embeddings.shape}")  # e.g., (N, 384)

    # 4) Save embeddings
    np.save(OUTPUT_EMB, embeddings)
    print(f"Saved embeddings to {OUTPUT_EMB}")

    # 5) Save metadata (aligned with embeddings row order)
    meta = pd.DataFrame({
        "contract_id": contract_ids,
        "chunk_id": chunk_ids,
        "text": raw_texts
    })
    meta.to_csv(OUTPUT_META, index=False)
    print(f"Saved metadata to {OUTPUT_META}")

    # 6) Optional: Save a small config file
    config = {
        "model_name": MODEL_NAME,
        "normalize_embeddings": NORMALIZE,
        "batch_size": BATCH_SIZE,
        "n_chunks": len(texts),
        "embedding_dim": int(embeddings.shape[1])
    }
    with open(os.path.join("data", "embed_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("Saved embed_config.json")

if __name__ == "__main__":
    main()
