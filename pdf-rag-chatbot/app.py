from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env
import streamlit as st
from utils import extract_pdf_text_with_metadata, chunk_documents
from rag_pipeline import RAGPipeline, generate_answer,  together_llm
from retriever import HybridRetriever
from evaluator import timed, log_trace_csv
import tempfile
import os

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF RAG Chatbot")

st.sidebar.header("Upload up to 2 PDFs")
uploaded_files = st.sidebar.file_uploader("Select two PDFs", type=["pdf"], accept_multiple_files=True)
use_reranker = st.sidebar.checkbox("Enable cross-encoder reranker (slower, better)", value=False)
beta = st.sidebar.slider("Fusion beta (semantic weight)", 0.0, 1.0, 0.7, 0.05)
k_bm25 = st.sidebar.slider("BM25 top-k", 5, 50, 15, 1)
k_dense = st.sidebar.slider("Dense top-k", 5, 50, 15, 1)
k_final = st.sidebar.slider("Final top-k (context docs)", 2, 10, 5, 1)

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None
if "hybrid" not in st.session_state:
    st.session_state.hybrid = None

def build_index_from_files(files):
    temp_paths = []
    for f in files:
        suffix = ".pdf"
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tf.write(f.read())
        tf.flush()
        temp_paths.append(tf.name)
    all_chunks = []
    for p in temp_paths:
        pages = extract_pdf_text_with_metadata(p)
        chunks = chunk_documents(pages, max_chars=800, overlap=120)
        all_chunks.extend(chunks)
    texts = [c["text"] for c in all_chunks]
    metas = [c["metadata"] for c in all_chunks]
    st.session_state.hybrid = HybridRetriever(texts, metas, embed_model_name="all-MiniLM-L6-v2",
                                              reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2" if use_reranker else None)

st.markdown("1) Upload PDFs in the sidebar. 2) Click ‘Build Index’. 3) Ask a question.")
if st.sidebar.button("Build Index"):
    if not uploaded_files or len(uploaded_files) == 0:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Indexing..."):
            build_index_from_files(uploaded_files)
        st.success("Index built!")

query = st.text_input("Ask a question about your PDFs:")
if st.button("Ask") and query:
    # If index not built yet, but PDFs are uploaded, build it on-demand
    if not st.session_state.hybrid:
        if uploaded_files and len(uploaded_files) > 0:
            with st.spinner("Indexing..."):
                build_index_from_files(uploaded_files)
            st.success("Index built!")
        else:
            st.warning("Please upload at least one PDF and click 'Build Index'.")
            st.stop()

    with st.spinner("Retrieving..."):
        results, t_retrieval = timed(st.session_state.hybrid.search)(
            query, k_bm25=k_bm25, k_dense=k_dense, k_final=k_final, beta=beta, use_reranker=use_reranker
        )
    # Build context
    ctx = []
    for idx, score in results:
        txt = st.session_state.hybrid.texts[idx]
        md = st.session_state.hybrid.metadatas[idx]
        ctx.append((txt, md))

    # Generation
    with st.spinner("Generating answer..."):
        (answer, prompt), t_gen = timed(generate_answer)( together_llm, query, ctx, max_context_docs=k_final)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for i, (_, md) in enumerate(ctx[:k_final], start=1):
        st.write(f"[DOC {i}] {md.get('source','')} \u2014 page {md.get('page','')}")

    # Logging
    log_trace_csv("traces.csv", {
        "query": query,
        "retrieval_time_s": round(t_retrieval,3),
        "generation_time_s": round(t_gen,3),
        "k_bm25": k_bm25,
        "k_dense": k_dense,
        "k_final": k_final,
        "beta": beta,
        "use_reranker": use_reranker
    })

st.caption("Tip: Enable reranker for tougher queries. Configure an external LLM in rag_pipeline.py for full answers.")
