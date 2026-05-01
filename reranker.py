"""
Cross-Encoder Re-Ranker:
Evaluates (query, document) pairs jointly for precise relevance scoring.
Significantly improves ranking over bi-encoder retrieval.
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReRanker:
    """
    Uses a cross-encoder transformer to score (query, doc) pairs.
    Much more accurate than bi-encoder similarity but slower,
    so applied only on the top-k candidates from retrieval.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Re-rank candidate documents using the cross-encoder.

        Args:
            query:      The user's question.
            candidates: List of chunk dicts from the hybrid retriever.

        Returns:
            Top-k re-ranked docs with updated 'score' from cross-encoder.
        """
        if not candidates:
            return []

        self._load_model()

        # Build (query, doc_text) pairs for the cross-encoder
        pairs = [(query, doc["text"]) for doc in candidates]

        scores = self._model.predict(pairs, batch_size=self.batch_size)

        # Attach cross-encoder scores and sort
        for doc, score in zip(candidates, scores):
            doc["ce_score"] = float(score)

        reranked = sorted(candidates, key=lambda d: d["ce_score"], reverse=True)

        # Use CE score as the primary score
        for doc in reranked:
            doc["score"] = doc["ce_score"]

        return reranked[: self.top_k]
