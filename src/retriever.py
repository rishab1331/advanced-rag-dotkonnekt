import os
import re
import glob
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.config import Config

@dataclass
class Chunk:
    chunk_id: str      # e.g., "07_hybridserach.txt#0"
    source_file: str   # e.g., "07_hybridserach.txt"
    text: str

def tokenize(text: str) -> List[str]:
    """Extracts lowercase alphanumeric tokens for BM25 keyword matching."""
    return re.findall(r"[a-z0-9]+", text.lower())

class HybridRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None

    def load_and_index(self, docs_dir: str = Config.DOCS_DIR) -> int:
        """Loads all .txt files from documents/, chunks them, and builds Vector + BM25 indexes."""
        txt_files = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in '{docs_dir}/'")

        self.chunks = []
        step = max(1, Config.CHUNK_SIZE_WORDS - Config.CHUNK_OVERLAP_WORDS)

        for filepath in txt_files:
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                words = f.read().split()

            if not words:
                continue

            chunk_idx = 0
            for start in range(0, len(words), step):
                slice_words = words[start : start + Config.CHUNK_SIZE_WORDS]
                chunk_text = " ".join(slice_words)
                self.chunks.append(
                    Chunk(
                        chunk_id=f"{filename}#{chunk_idx}",
                        source_file=filename,
                        text=chunk_text,
                    )
                )
                chunk_idx += 1
                if start + Config.CHUNK_SIZE_WORDS >= len(words):
                    break

        # 1. Build BM25 Keyword Index
        tokenized_corpus = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Build Dense Vector Index (Normalized for Cosine Similarity)
        texts = [c.text for c in self.chunks]
        vecs = self.embedder.encode(texts, normalize_embeddings=True)
        self.embeddings = np.array(vecs, dtype=np.float32)

        return len(self.chunks)

    def search_single_query(self, query: str) -> Dict[str, Dict[str, Any]]:
        """Runs Vector + BM25 search for one query and fuses rankings via RRF."""
        assert self.embeddings is not None and self.bm25 is not None, "Call load_and_index() first."

        # Vector Search (Cosine Similarity)
        q_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        vector_scores = np.dot(self.embeddings, q_vec)
        top_vec_indices = np.argsort(vector_scores)[::-1][:Config.TOP_K_VECTOR]

        # BM25 Keyword Search
        bm25_scores = self.bm25.get_scores(tokenize(query))
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:Config.TOP_K_BM25]

        # Reciprocal Rank Fusion (RRF)
        fused: Dict[str, Dict[str, Any]] = {}

        for rank, idx in enumerate(top_vec_indices, start=1):
            c = self.chunks[int(idx)]
            fused[c.chunk_id] = {
                "chunk": c,
                "rrf_score": 1.0 / (Config.RRF_K + rank),
                "vector_score": round(float(vector_scores[int(idx)]), 4),
                "bm25_score": round(float(bm25_scores[int(idx)]), 4),
            }

        for rank, idx in enumerate(top_bm25_indices, start=1):
            c = self.chunks[int(idx)]
            if c.chunk_id not in fused:
                fused[c.chunk_id] = {
                    "chunk": c,
                    "rrf_score": 0.0,
                    "vector_score": round(float(vector_scores[int(idx)]), 4),
                    "bm25_score": round(float(bm25_scores[int(idx)]), 4),
                }
            fused[c.chunk_id]["rrf_score"] += 1.0 / (Config.RRF_K + rank)

        return fused

    def search_multi_queries(self, queries: List[str], top_n: int = 8) -> List[Dict[str, Any]]:
        """Runs Hybrid RRF search across multiple decomposed sub-queries."""
        combined: Dict[str, Dict[str, Any]] = {}
        for q in queries:
            results = self.search_single_query(q)
            for chunk_id, data in results.items():
                if chunk_id not in combined:
                    combined[chunk_id] = data
                else:
                    combined[chunk_id]["rrf_score"] += data["rrf_score"]

        ranked = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
        return ranked[:top_n]