# app_gradio.py
import gradio as gr
from utils import extract_pdf_text_with_metadata, chunk_documents
from retriever import HybridRetriever
from rag_pipeline import generate_answer, together_llm  # make sure together_llm reads TOGETHER_API_KEY

hybrid = None

def build_index(file_objs, use_reranker=False, beta=0.7, k_bm25=15, k_dense=15, k_final=5):
    global hybrid
    if not file_objs:
        return "Upload at least one PDF before building the index."
    all_chunks = []
    for fo in file_objs:
        pages = extract_pdf_text_with_metadata(fo.name)
        chunks = chunk_documents(pages, max_chars=800, overlap=120)
        all_chunks.extend(chunks)
    texts = [c["text"] for c in all_chunks]
    metas = [c["metadata"] for c in all_chunks]
    hybrid = HybridRetriever(
        texts, metas,
        embed_model_name="all-MiniLM-L6-v2",
        reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2" if use_reranker else None
    )
    return f"Index built with {len(texts)} chunks."

def ask(query, use_reranker=False, beta=0.7, k_bm25=15, k_dense=15, k_final=5):
    global hybrid
    if hybrid is None:
        return "Please build the index first.", ""
    results = hybrid.search(query, k_bm25=k_bm25, k_dense=k_dense, k_final=k_final, beta=beta, use_reranker=use_reranker)
    ctx = []
    for idx, score in results:
        txt = hybrid.texts[idx]
        md = hybrid.metadatas[idx]
        ctx.append((txt, md))
    answer, _ = generate_answer(together_llm, query, ctx, max_context_docs=k_final)
    sources = "\n".join([f"[DOC {i}] {md.get('source','')} — page {md.get('page','')}"
                         for i, (_, md) in enumerate(ctx[:k_final], start=1)])
    return answer, sources

with gr.Blocks(title="PDF RAG Chatbot (Gradio)") as demo:
    gr.Markdown("## 📄 PDF RAG Chatbot\nUpload up to two PDFs, build the index, and ask questions.")
    with gr.Row():
        files = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
        use_reranker = gr.Checkbox(label="Enable reranker (slower, better)", value=False)
    with gr.Row():
        beta = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Fusion beta (semantic weight)")
        k_bm25 = gr.Slider(5, 50, value=15, step=1, label="BM25 top-k")
        k_dense = gr.Slider(5, 50, value=15, step=1, label="Dense top-k")
        k_final = gr.Slider(2, 10, value=5, step=1, label="Final top-k (context docs)")
    build_btn = gr.Button("Build Index")
    status = gr.Textbox(label="Status", interactive=False)
    build_btn.click(build_index, inputs=[files, use_reranker, beta, k_bm25, k_dense, k_final], outputs=status)

    query = gr.Textbox(label="Ask a question")
    ask_btn = gr.Button("Ask")
    answer = gr.Markdown()
    sources = gr.Textbox(label="Sources", lines=4)
    ask_btn.click(ask, inputs=[query, use_reranker, beta, k_bm25, k_dense, k_final], outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch()
