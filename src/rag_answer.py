# src/rag_answer.py
from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class RAGConfig:
    # Generous defaults while debugging
    max_context_chars: int = 20000    # try 16,000–24,000
    per_chunk_limit: int = 1200       # try 1,200–2,000
    min_score: Optional[float] = None # disable filtering while debugging
    system_prompt: str = (
        "You are a careful legal assistant. Answer ONLY using the provided Context.\n"
        "If the answer is not in the Context, reply exactly: 'Not found in the provided documents.'\n"
        "Keep answers concise and include direct quotes when appropriate."
    )
    sources_prefix: str = "Sources: "


class RAGAnswerer:
    """
    Retrieves top-k chunks, builds a bounded context, and asks the LLM.
    Includes verbose debug logs to diagnose 'Not found...' when Search shows hits.
    """

    def __init__(self, retriever, cfg: Optional[RAGConfig] = None, k: int = 8):
        self.retriever = retriever
        self.cfg = cfg or RAGConfig()
        self.k_default = int(k)

    def answer(
        self,
        query: str,
        llm: Callable[..., str],
        k: Optional[int] = None,
        temperature: float = 0.2,
    ) -> Dict[str, object]:
        try:
            top_k = int(k) if k is not None else self.k_default

            # 1) Retrieve
            hits = self._safe_retrieve(query, top_k)
            print("DEBUG ask top_k:", top_k, "raw_hits:", len(hits))

            # 2) Filter/clean (very permissive while debugging)
            hits_filtered = self._filter_hits(hits, self.cfg.min_score)
            print("DEBUG hits_filtered:", len(hits_filtered))

            # 3) Sort by score desc (None scores go last)
            hits_sorted = self._sort_by_score_desc(hits_filtered)

            # 4) Build bounded context
            context_str, used_hits = self._build_context(hits_sorted)
            print("DEBUG used_hits:", len(used_hits), "context_len:", len(context_str))
            if used_hits:
                first_snip = (used_hits[0].get("text") or "")[:200].replace("\n", " ")
                print("DEBUG first_snippet:", first_snip)

            # 5) Build prompt and show head
            prompt = self._build_prompt(query, context_str)
            print("DEBUG prompt_head:\n", prompt[:350])

            # 6) LLM call (pass temperature if supported)
            answer = llm_call_with_params(llm, prompt, temperature)

            if not answer or not answer.strip():
                answer = "Not found in the provided documents."

            sources_text = self._format_sources(used_hits)
            return {
                "answer": answer.strip(),
                "sources_text": sources_text,
                "hits": used_hits,
            }

        except Exception as e:
            print("RAG_ANSWER ERROR:", repr(e))
            traceback.print_exc()
            return {
                "answer": "Error. See server logs for details.",
                "sources_text": "",
                "hits": [],
            }

    # ----------------------
    # Internals
    # ----------------------

    def _safe_retrieve(self, query: str, k: int) -> List[Dict]:
        try:
            if not self.retriever:
                print("DEBUG retriever is None")
                return []
            q = (query or "").strip()
            return self.retriever.search(q, k=int(k)) or []
        except Exception as e:
            print("RETRIEVE ERROR:", repr(e))
            traceback.print_exc()
            return []

    def _filter_hits(self, hits: List[Dict], min_score: Optional[float]) -> List[Dict]:
        if not hits:
            return []
        cleaned: List[Dict] = []
        for h in hits:
            if not isinstance(h, dict):
                continue
            text = (h.get("text") or "").strip()
            if not text:
                continue
            # Safe score conversion
            sv = h.get("score")
            try:
                score = float(sv) if sv is not None else None
            except Exception:
                score = None
            # Optional threshold (disabled if None)
            if min_score is not None and score is not None and score < float(min_score):
                continue
            cleaned.append({
                **h,
                "text": text,
                "score": score,
                "chunk_id": h.get("chunk_id"),
                "doc_id": h.get("doc_id") or h.get("contract_id"),
            })
        return cleaned

    def _sort_by_score_desc(self, hits: List[Dict]) -> List[Dict]:
        if not hits:
            return []
        def keyfn(h: Dict):
            s = h.get("score")
            # Put scored hits first (higher score first), then None scores
            return (0, -float(s)) if isinstance(s, (int, float)) else (1, 0)
        try:
            return sorted(hits, key=keyfn)
        except Exception:
            return hits

    def _build_context(self, hits: List[Dict]) -> Tuple[str, List[Dict]]:
        if not hits:
            return "", []
        remaining = max(0, int(self.cfg.max_context_chars))
        per_limit = max(0, int(self.cfg.per_chunk_limit))
        parts: List[str] = []
        used: List[Dict] = []
        for h in hits:
            raw = (h.get("text") or "").strip()
            if not raw:
                continue
            block = raw[:per_limit].strip()
            to_add = f"\n---\n{block}\n"
            if len(to_add) <= remaining:
                parts.append(to_add)
                used.append(h)
                remaining -= len(to_add)
            else:
                break
        return "".join(parts).strip(), used

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            f"Question: {(query or '').strip()}\n\n"
            f"Context:\n{context if context else '[No relevant context retrieved]'}\n\n"
            "Instructions:\n"
            "- Answer ONLY using the Context.\n"
            "- If the answer is not present, reply exactly: 'Not found in the provided documents.'\n"
            "- Be concise and include short quotes when helpful."
        )

    def _format_sources(self, hits: List[Dict]) -> str:
        if not hits:
            return self.cfg.sources_prefix
        labels: List[str] = []
        for h in hits:
            cid = h.get("chunk_id")
            did = h.get("doc_id")
            if cid and did:
                labels.append(f"{did}:{cid}")
            elif cid:
                labels.append(str(cid))
            elif did:
                labels.append(str(did))
        return f"{self.cfg.sources_prefix}" + (", ".join(labels) if labels else "")


def llm_call_with_params(llm: Callable[..., str], prompt: str, temperature: float) -> str:
    try:
        return llm(prompt, temperature=float(temperature))  # type: ignore
    except TypeError:
        return llm(prompt)
