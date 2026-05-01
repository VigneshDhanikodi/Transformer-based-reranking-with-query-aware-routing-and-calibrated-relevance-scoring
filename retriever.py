"""
Hybrid Retriever: Combines BM25 sparse retrieval with FAISS dense retrieval.
Scores are fused using Reciprocal Rank Fusion (RRF).
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines:
      - BM25  → keyword/lexical matching  (rank-k list)
      - FAISS → semantic/dense matching   (rank-k list)
    Fuses results via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        top_k: int = 20,
        rrf_k: int = 60,
        alpha: float = 0.5,
    ):
        self.embedding_model_name = embedding_model
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.top_k = top_k
        self.rrf_k = rrf_k          # RRF constant (higher → smoother fusion)
        self.alpha = alpha           # weight for dense vs sparse (0=sparse, 1=dense)

        self._corpus: List[Dict] = []
        self._bm25 = None
        self._faiss_index = None
        self._embedder = None

    # ------------------------------------------------------------------ #
    #  Indexing                                                            #
    # ------------------------------------------------------------------ #

    def index(self, chunks: List[Dict]) -> None:
        """Build BM25 and FAISS indexes from processed chunks."""
        self._corpus = chunks
        texts = [c["text"] for c in chunks]

        logger.info("Building BM25 index...")
        self._build_bm25(texts)

        logger.info("Building FAISS index...")
        self._build_faiss(texts)

        logger.info(f"Indexes built for {len(chunks)} chunks.")

    def _build_bm25(self, texts: List[str]) -> None:
        from rank_bm25 import BM25Okapi
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized, k1=self.bm25_k1, b=self.bm25_b)

    def _build_faiss(self, texts: List[str]) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.embedding_model_name)
        embeddings = self._embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)   # Inner-product = cosine (normalized)
        self._faiss_index.add(embeddings)

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve top-k candidates by fusing BM25 and FAISS scores.

        Returns list of chunk dicts, each with an added 'score' key.
        """
        bm25_ranks = self._bm25_retrieve(query)
        faiss_ranks = self._faiss_retrieve(query)
        fused = self._reciprocal_rank_fusion(bm25_ranks, faiss_ranks)
        return fused[:self.top_k]

    def _bm25_retrieve(self, query: str) -> List[Tuple[int, float]]:
        """Return (chunk_idx, score) list sorted descending."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:self.top_k * 2]

    def _faiss_retrieve(self, query: str) -> List[Tuple[int, float]]:
        """Return (chunk_idx, score) list sorted descending."""
        import numpy as np
        q_emb = self._embedder.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")
        scores, indices = self._faiss_index.search(q_emb, self.top_k * 2)
        return list(zip(indices[0].tolist(), scores[0].tolist()))

    def _reciprocal_rank_fusion(
        self,
        bm25_ranks: List[Tuple[int, float]],
        faiss_ranks: List[Tuple[int, float]],
    ) -> List[Dict]:
        """
        Fuse two ranked lists via RRF.
        score(d) = Σ 1 / (k + rank(d))
        """
        rrf_scores: Dict[int, float] = {}

        for rank, (idx, _) in enumerate(bm25_ranks, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1 - self.alpha) / (self.rrf_k + rank)

        for rank, (idx, _) in enumerate(faiss_ranks, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + self.alpha / (self.rrf_k + rank)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        results = []
        for idx in sorted_ids:
            if idx < len(self._corpus):
                doc = dict(self._corpus[idx])
                doc["score"] = round(rrf_scores[idx], 6)
                results.append(doc)
        return results
