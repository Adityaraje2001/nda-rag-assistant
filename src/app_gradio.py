# src/app_gradio.py
import gradio as gr
import pandas as pd
from retrieve import Retriever

retriever = Retriever(k=5)

def search_ui(query, top_k):
    results = retriever.search(query, k=int(top_k))
    if not results:
        return pd.DataFrame(columns=["rank", "score", "contract_id", "chunk_id", "text"])
    df = pd.DataFrame(results)[["rank", "score", "contract_id", "chunk_id", "text"]]
    df["score"] = df["score"].round(4)
    return df

with gr.Blocks() as demo:
    gr.Markdown("## Contract Retrieval Demo")
    with gr.Row():
        query = gr.Textbox(label="Query", placeholder="e.g., Termination rights or Governing law")
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top K")
    results = gr.Dataframe(
        headers=["rank", "score", "contract_id", "chunk_id", "text"],
        datatype=["number", "number", "str", "str", "str"],
        row_count=(5, "dynamic"),
        col_count=(5, "fixed"),
        wrap=True,
        interactive=False,
        label="Results"
    )
    btn = gr.Button("Search")
    btn.click(fn=search_ui, inputs=[query, top_k], outputs=results)

if __name__ == "__main__":
    demo.launch()
