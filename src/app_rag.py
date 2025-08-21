# src/app_rag.py
from __future__ import annotations

import gradio as gr
import traceback

from .retrieve import Retriever
from .rag_answer import RAGAnswerer, RAGConfig
from .llm_provider import llm_call

# Instantiate once at import
try:
    retriever = Retriever(k=8)
except Exception as e:
    print("RETRIEVER INIT ERROR:", repr(e))
    traceback.print_exc()
    # Defer failure to runtime interactions
    retriever = None

rag_cfg = RAGConfig(
    max_context_chars=16000,
    per_chunk_limit=1800,
    min_score=None,  # disable filtering while testing
)
rag = RAGAnswerer(retriever, cfg=rag_cfg, k=8) if retriever else None


# ---------------------
# Gradio handlers
# ---------------------
def on_search(query: str, top_k: int):
    try:
        if not retriever:
            return [["-", "-", "Retriever failed to initialize. See server logs."]]
        q = (query or "").strip()
        if not q:
            return [["-", "-", "Enter a search query."]]
        hits = retriever.search(q, k=int(top_k or 8)) or []
        rows = []
        for i, h in enumerate(hits, start=1):
            score = h.get("score")
            txt = (h.get("text") or "")[:200].replace("\n", " ")
            rows.append([i, f"{score:.4f}" if score is not None else "-", txt])
        if not rows:
            rows = [["-", "-", "No hits"]]
        return rows
    except Exception as e:
        print("SEARCH ERROR:", repr(e))
        traceback.print_exc()
        return [["-", "-", f"Error: {e!r}"]]


def on_ask(query: str, top_k: int, temperature: float):
    try:
        if not rag:
            return "Error. RAG not initialized. See server logs.", ""
        q = (query or "").strip()
        if not q:
            return "Please enter a question.", ""
        out = rag.answer(q, llm=llm_call, k=int(top_k or 8), temperature=float(temperature or 0.2))
        return out.get("answer", ""), out.get("sources_text", "")
    except Exception as e:
        print("ASK ERROR:", repr(e))
        traceback.print_exc()
        return "Error. See server logs for details.", ""


# ---------------------
# Gradio UI (module-scope `demo`)
# ---------------------
with gr.Blocks(title="RAG Ask") as demo:
    gr.Markdown("## RAG Playground")

    with gr.Tab("Search"):
        gr.Markdown("Retrieve top-k chunks to verify your index.")
        with gr.Row():
            s_query = gr.Textbox(label="Query", value="governing law")
            s_k = gr.Slider(1, 15, value=8, step=1, label="Top K")
        s_btn = gr.Button("Search")
        s_table = gr.Dataframe(headers=["Rank", "Score", "Snippet"], datatype=["number", "str", "str"], row_count=5, interactive=False)
        s_btn.click(fn=on_search, inputs=[s_query, s_k], outputs=[s_table])

    with gr.Tab("Ask (RAG)"):
        with gr.Row():
            a_query = gr.Textbox(label="Question", value="Termination rights")
            a_btn = gr.Button("Ask")
        a_answer = gr.Markdown(label="Answer")
        a_sources = gr.Markdown(label="Sources")
        with gr.Row():
            a_k = gr.Slider(1, 15, value=8, step=1, label="Top K")
            a_temp = gr.Slider(0.0, 1.0, value=0.2, step=0.1, label="Temperature")
        a_btn.click(fn=on_ask, inputs=[a_query, a_k, a_temp], outputs=[a_answer, a_sources])


# Launch only when running this module directly
if __name__ == "__main__":
    demo.launch()
