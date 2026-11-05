"""Rerank retrieved chunks with a cross-encoder (second stage of two-stage retrieval)."""
from sentence_transformers import CrossEncoder

# A small, fast reranker trained on search data (MS MARCO). Larger alternative to try
# later: "BAAI/bge-reranker-base" (more accurate, but about 10x bigger and slower).
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker() -> CrossEncoder:
    """Load the reranker model (downloaded automatically the first time)."""
    return CrossEncoder(RERANKER_NAME)


def rerank(query: str, chunks: list[dict], reranker: CrossEncoder, top_k: int) -> list[dict]:
    """Score each (question, chunk) pair together and return the top_k best chunks."""
    pairs = [(query, c["content"]) for c in chunks]     # the reranker reads both texts at once
    scores = reranker.predict(pairs)                     # one relevance score per pair
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)             # keep it, useful for inspection
    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)   # best first
    return ranked[:top_k]