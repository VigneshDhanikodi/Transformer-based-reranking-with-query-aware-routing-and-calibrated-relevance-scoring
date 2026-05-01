"""
Hybrid RAG Pipeline for Medical Question Answering
Combines BM25 + FAISS retrieval with Cross-Encoder re-ranking and LLM generation.
"""

import os
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .retriever import HybridRetriever
from .reranker import CrossEncoderReRanker
from .generator import LLMGenerator
from .document_processor import DocumentProcessor
from .source_scorer import SourceAwareScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]
    confidence: float
    latency_ms: float
    retrieval_scores: List[float]


class MedicalRAGPipeline:
    """
    End-to-end Hybrid RAG pipeline for medical question answering.

    Architecture:
        Query → Hybrid Retrieval (BM25 + FAISS)
              → Source-Aware Scoring
              → Cross-Encoder Re-Ranking
              → LLM Answer Generation
              → Response with Citations
    """

    def __init__(self, config: Dict):
        self.config = config
        self.processor = DocumentProcessor(
            chunk_size=config.get("chunk_size", 512),
            chunk_overlap=config.get("chunk_overlap", 64),
        )
        self.retriever = HybridRetriever(
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            bm25_k1=config.get("bm25_k1", 1.5),
            bm25_b=config.get("bm25_b", 0.75),
            top_k=config.get("retrieval_top_k", 20),
        )
        self.source_scorer = SourceAwareScorer(
            trusted_sources=config.get("trusted_sources", ["WHO", "NIH", "PubMed"])
        )
        self.reranker = CrossEncoderReRanker(
            model_name=config.get(
                "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            top_k=config.get("rerank_top_k", 5),
        )
        self.generator = LLMGenerator(
            model_name=config.get("llm_model", "google/flan-t5-large"),
            max_new_tokens=config.get("max_new_tokens", 512),
        )
        self._is_indexed = False

    def index_documents(self, documents: List[Dict]) -> None:
        """
        Process and index documents into BM25 + FAISS stores.

        Args:
            documents: List of dicts with keys: 'text', 'source', 'metadata'
        """
        logger.info(f"Indexing {len(documents)} documents...")
        chunks = self.processor.process(documents)
        self.retriever.index(chunks)
        self._is_indexed = True
        logger.info(f"Indexed {len(chunks)} chunks successfully.")

    def query(self, question: str, verbose: bool = False) -> RAGResponse:
        """
        Run the full RAG pipeline on a medical question.

        Args:
            question: User's medical question
            verbose:  Whether to log intermediate steps

        Returns:
            RAGResponse with answer, sources, confidence, and latency
        """
        if not self._is_indexed:
            raise RuntimeError("No documents indexed. Call index_documents() first.")

        start = time.time()

        # Step 1: Hybrid retrieval
        candidates = self.retriever.retrieve(question)
        if verbose:
            logger.info(f"Retrieved {len(candidates)} candidates")

        # Step 2: Source-aware boosting
        candidates = self.source_scorer.boost(candidates)

        # Step 3: Cross-encoder re-ranking
        top_docs = self.reranker.rerank(question, candidates)
        if verbose:
            logger.info(f"Top {len(top_docs)} docs after re-ranking")

        # Step 4: LLM answer generation
        context = self._build_context(top_docs)
        answer = self.generator.generate(question, context)

        # Step 5: Confidence estimation
        confidence = self._estimate_confidence(top_docs)

        latency_ms = (time.time() - start) * 1000

        return RAGResponse(
            answer=answer,
            sources=[
                {
                    "text": d["text"][:200],
                    "source": d.get("source", "Unknown"),
                    "score": d.get("score", 0.0),
                }
                for d in top_docs
            ],
            confidence=confidence,
            latency_ms=latency_ms,
            retrieval_scores=[d.get("score", 0.0) for d in top_docs],
        )

    def _build_context(self, docs: List[Dict]) -> str:
        """Concatenate top documents into a context string."""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("source", "Unknown")
            parts.append(f"[Source {i} - {source}]\n{doc['text']}\n")
        return "\n".join(parts)

    def _estimate_confidence(self, docs: List[Dict]) -> float:
        """Estimate confidence from top re-ranking scores (0.0–1.0)."""
        if not docs:
            return 0.0
        scores = [d.get("score", 0.0) for d in docs[:3]]
        avg = sum(scores) / len(scores)
        # Normalize cross-encoder scores (typically -10 to 10) to [0, 1]
        normalized = 1 / (1 + pow(2.718, -avg))
        return round(normalized, 4)
