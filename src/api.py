# src/api.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

from .retrieve import Retriever
from .rag_answer import RAGAnswerer
from .llm_provider import llm_call

# Config
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = os.getenv("MIN_SCORE")
MIN_SCORE = float(MIN_SCORE) if MIN_SCORE else None

retriever = Retriever(k=TOP_K)
rag = RAGAnswerer(retriever, k=TOP_K, min_score=MIN_SCORE)

app = FastAPI(title="Contract RAG API", version="1.0.0")

class SearchRequest(BaseModel):
    query: str
    k: int = TOP_K

class AskRequest(BaseModel):
    question: str
    k: int = TOP_K

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search")
def search(req: SearchRequest):
    hits = retriever.search(req.query, k=req.k)
    return {"hits": hits}

@app.post("/ask")
def ask(req: AskRequest):
    out = rag.answer(req.question, llm=llm_call, k=req.k)
    return out

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)
