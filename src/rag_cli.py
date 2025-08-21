# src/rag_cli.py
import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import argparse
import textwrap

from .retrieve import Retriever
from .rag_answer import RAGAnswerer
from .llm_provider import llm_call

def main():
    parser = argparse.ArgumentParser(
        description="CLI tester for RAG: retrieves top-k chunks and generates an answer with citations."
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Question to ask (e.g., 'What is the governing law?'). If omitted, runs sample questions."
    )
    parser.add_argument(
        "-k", "--top_k",
        type=int,
        default=5,
        help="Top-k chunks to retrieve (default: 5)"
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=None,
        help="Optional minimum similarity score threshold to filter hits (e.g., 0.25)"
    )
    args = parser.parse_args()

    retriever = Retriever(k=args.top_k)
    rag = RAGAnswerer(
        retriever,
        k=args.top_k,
        max_context_chars=12000,
        per_chunk_limit=1500,
        min_score=args.min_score
    )

    def run_one(question: str):
        print("\n" + "="*80)
        print(f"Question: {question}")
        out = rag.answer(question, llm=llm_call, k=args.top_k)

        # Answer
        print("\nAnswer:\n")
        print(textwrap.fill(out["answer"], width=100))

        # Top hits (preview)
        hits = out["hits"]
        if not hits:
            print("\n(No hits returned)")
            return

        print("\nTop hits:")
        for h in hits[:min(5, len(hits))]:
            preview = (h.get("text") or "")[:160].replace("\n", " ")
            print(f"- rank={h.get('rank')} score={h.get('score'):.4f} "
                  f"contract_id={h.get('contract_id')} chunk_id={h.get('chunk_id')}")
            print(f"  {preview}...")
        print("="*80 + "\n")

    if args.question:
        run_one(args.question)
    else:
        samples = [
            "What is the governing law?",
            "Termination rights",
            "Confidentiality obligations",
        ]
        for q in samples:
            run_one(q)

if __name__ == "__main__":
    main()
