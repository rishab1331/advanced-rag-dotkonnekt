from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from src.config import Config

class CrossEncoderReranker:
    """Re-ranks hybrid search candidates using a Cross-Encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = Config.RERANK_TOP_K,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        # Create (query, chunk_text) pairs for joint cross-attention scoring
        pairs = [[query, item["chunk"].text] for item in candidates]
        scores = self.model.predict(pairs)

        reranked_list: List[Dict[str, Any]] = []
        for item, score in zip(candidates, scores):
            chunk = item["chunk"]
            reranked_list.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "text": chunk.text,
                    "rrf_score": round(float(item["rrf_score"]), 5),
                    "vector_score": item["vector_score"],
                    "bm25_score": item["bm25_score"],
                    "rerank_score": round(float(score), 4),
                }
            )

        # Sort by Cross-Encoder score descending
        reranked_list.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_list[:top_k]