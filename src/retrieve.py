import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json

class SimpleVectorStore:
    def __init__(self, embeddings, metadata):
        self.embeddings = embeddings
        self.metadata = metadata
        
    def search(self, query_embedding, k=20):
        # Compute cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                "rank": rank,
                "score": float(similarities[idx]),
                "doc_id": self.metadata.get(idx, {}).get("doc_id", ""),
                "contract_id": self.metadata.get(idx, {}).get("doc_id", ""),
                "chunk_id": self.metadata.get(idx, {}).get("chunk_id", str(idx)),
                "text": self.metadata.get(idx, {}).get("text", ""),
                "title": self.metadata.get(idx, {}).get("title", "")
            })
        return results

# In your retrieve.py, replace FAISS loading with:
def load_simple_index():
    embeddings = np.load("data/embeddings.npy")
    with open("data/metadata.json", "r") as f:  # Convert your metadata to JSON
        metadata = json.load(f)
    return SimpleVectorStore(embeddings, metadata)
