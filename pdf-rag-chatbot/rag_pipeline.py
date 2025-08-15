# rag_pipeline.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from prompt import build_augmented_prompt
import time

class RAGPipeline:
    def __init__(self, collection_name="pdf_chunks", embed_model_name="all-MiniLM-L6-v2"):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = SentenceTransformer(embed_model_name)

    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(self.collection.name)

    def add_documents(self, docs: List[Dict]):
        texts = [d["text"] for d in docs]
        metadatas = [d["metadata"] for d in docs]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True).tolist()
        ids = [f'{m["source"]}-{m["page"]}-{m.get("chunk_id",0)}-{i}' for i,m in enumerate(metadatas)]
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def dense_query(self, query:str, top_k=10):
        q_emb = self.embedder.encode([query], convert_to_numpy=True).tolist()
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents","metadatas","distances","embeddings","ids"])
        return res

def generate_answer(llm_fn, query:str, contexts:List[Tuple[str,Dict]], max_context_docs=5):
    # contexts: list of (text, metadata)
    context_blocks = []
    for i, (txt, md) in enumerate(contexts[:max_context_docs], start=1):
        label = f"DOC {i}"
        source = f'{md.get("source","")}, page {md.get("page","")}'
        context_blocks.append((label, txt, source))
    prompt = build_augmented_prompt(query, context_blocks)
    return llm_fn(prompt), prompt

# rag_pipeline.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

def together_llm(prompt: str) -> str:
    """Call Together's chat completions API using the official SDK.

    - Reads model from TOGETHER_MODEL env var if set.
    - Falls back across a list of smaller, commonly accessible models on Free tier.
    - Returns a helpful message if all models are blocked.
    """
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        return "No TOGETHER_API_KEY set."

    # Import inside the function to avoid hard dependency at import time
    try:
        from together import Together
        from together.error import InvalidRequestError
    except Exception:
        # Older SDKs may not expose together.error; import client only
        try:
            from together import Together  # type: ignore
            InvalidRequestError = Exception  # type: ignore
        except Exception as e:  # pragma: no cover
            return f"Together SDK not installed: {e}"

    client = Together()  # auth defaults to os.environ['TOGETHER_API_KEY']

    # Prefer explicit model via env; otherwise try a safe fallback list.
    configured = os.getenv("TOGETHER_MODEL")
    fallback_models = [
        # Smaller, widely available chat models
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "google/gemma-7b-it",
        "google/gemma-2b-it",
        "openchat/openchat-3.5-1210",
        "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        # Very small Together-hosted options
        "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
    ]

    candidates = [configured] if configured else []
    candidates += [m for m in fallback_models if m != configured]

    last_err = None
    for model_name in candidates:
        if not model_name:
            continue
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except InvalidRequestError as e:
            # Likely permission/tier issue (403) -> try next model
            last_err = e
            continue
        except Exception as e:  # network or other API errors
            last_err = e
            continue

    return (
        "All Together models attempted are unavailable on your current tier. "
        "Set TOGETHER_MODEL to a model your account can access (e.g., 'mistralai/Mistral-7B-Instruct-v0.2' "
        "or 'google/gemma-2b-it') or add credits. Last error: %s" % (last_err,)
    )

