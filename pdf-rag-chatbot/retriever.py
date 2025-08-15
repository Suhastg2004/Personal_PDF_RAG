# retriever.py
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from typing import List, Dict, Tuple

class HybridRetriever:
    def __init__(self, texts: List[str], metadatas: List[Dict], embed_model_name="all-MiniLM-L6-v2", reranker_name=None):
        self.texts = texts
        self.metadatas = metadatas
        self.tokenized_corpus = [t.split() for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.embedder = SentenceTransformer(embed_model_name)
        self.doc_embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        self.reranker = CrossEncoder(reranker_name) if reranker_name else None

    def search(self, query: str, k_bm25=15, k_dense=15, k_final=6, beta=0.7, use_reranker=False) -> List[Tuple[int, float]]:
        # BM25
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top = np.argsort(bm25_scores)[::-1][:k_bm25]

        # Dense
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        dense_scores = (self.doc_embeddings @ q_emb) / (np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-12)
        dense_top = np.argsort(dense_scores)[::-1][:k_dense]

        # RRF-like fusion with beta weighting
        # Convert ranks to reciprocal scores
        rrf = {}
        for rank, idx in enumerate(bm25_top):
            rrf[idx] = rrf.get(idx, 0) + (1-beta) * 1.0/(rank+1)
        for rank, idx in enumerate(dense_top):
            rrf[idx] = rrf.get(idx, 0) + beta * 1.0/(rank+1)

        fused_sorted = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:max(k_final, 2*k_final)]
        candidates = [i for i,_ in fused_sorted]

        if use_reranker and self.reranker:
            pairs = [(query, self.texts[i]) for i in candidates]
            rerank_scores = self.reranker.predict(pairs)
            order = np.argsort(rerank_scores)[::-1][:k_final]
            ranked = [(candidates[i], float(rerank_scores[i])) for i in order]
            return ranked
        else:
            return [(idx, float(rrf[idx])) for idx,_ in fused_sorted[:k_final]]
