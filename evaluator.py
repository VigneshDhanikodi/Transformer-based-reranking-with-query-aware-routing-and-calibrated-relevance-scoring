"""
Evaluation Module:
Computes ROUGE, BERTScore, retrieval precision, and latency metrics
for the Hybrid RAG medical QA system.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluates both generation quality and retrieval effectiveness.

    Metrics:
      Generation : ROUGE-1, ROUGE-2, ROUGE-L, BERTScore (F1)
      Retrieval  : Precision@K, Recall@K, MRR, NDCG@K
      System     : Average latency (ms)
    """

    def __init__(self, use_bertscore: bool = False):
        self.use_bertscore = use_bertscore
        self._rouge = None

    def _load_rouge(self):
        if self._rouge is None:
            from rouge_score import rouge_scorer
            self._rouge = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )

    # ------------------------------------------------------------------ #
    #  Generation Quality                                                  #
    # ------------------------------------------------------------------ #

    def rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
        self._load_rouge()
        scores = self._rouge.score(reference, prediction)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }

    def bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BERTScore F1 using microsoft/deberta-xlarge-mnli."""
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn(
                predictions, references, lang="en", verbose=False
            )
            return {
                "bertscore_precision": round(P.mean().item(), 4),
                "bertscore_recall":    round(R.mean().item(), 4),
                "bertscore_f1":        round(F1.mean().item(), 4),
            }
        except ImportError:
            logger.warning("bert_score not installed. pip install bert-score")
            return {}

    # ------------------------------------------------------------------ #
    #  Retrieval Quality                                                   #
    # ------------------------------------------------------------------ #

    def precision_at_k(
        self, retrieved_ids: List[str], relevant_ids: List[str], k: int = 5
    ) -> float:
        """Precision@K = |retrieved_top_k ∩ relevant| / K"""
        top_k = set(retrieved_ids[:k])
        relevant = set(relevant_ids)
        return round(len(top_k & relevant) / k, 4)

    def recall_at_k(
        self, retrieved_ids: List[str], relevant_ids: List[str], k: int = 5
    ) -> float:
        """Recall@K = |retrieved_top_k ∩ relevant| / |relevant|"""
        if not relevant_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        relevant = set(relevant_ids)
        return round(len(top_k & relevant) / len(relevant), 4)

    def mean_reciprocal_rank(
        self, retrieved_ids: List[str], relevant_ids: List[str]
    ) -> float:
        """MRR: 1/rank of first relevant document."""
        relevant = set(relevant_ids)
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant:
                return round(1.0 / rank, 4)
        return 0.0

    def ndcg_at_k(
        self, retrieved_ids: List[str], relevant_ids: List[str], k: int = 5
    ) -> float:
        """NDCG@K using binary relevance."""
        relevant = set(relevant_ids)
        gains = [1 if doc_id in relevant else 0 for doc_id in retrieved_ids[:k]]
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))

        ideal = sorted(gains, reverse=True)
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))

        return round(dcg / idcg if idcg > 0 else 0.0, 4)

    # ------------------------------------------------------------------ #
    #  Full Evaluation Suite                                               #
    # ------------------------------------------------------------------ #

    def evaluate_dataset(
        self,
        pipeline,
        test_cases: List[Dict],
        verbose: bool = True,
    ) -> Dict:
        """
        Run evaluation over a test set.

        Args:
            pipeline:   MedicalRAGPipeline instance
            test_cases: [{'question': str, 'answer': str, 'relevant_ids': list}, ...]
            verbose:    Print per-sample progress

        Returns:
            Dict of aggregated metrics.
        """
        all_rouge1, all_rouge2, all_rougeL = [], [], []
        all_precision, all_recall, all_mrr, all_ndcg = [], [], [], []
        latencies = []

        predictions, references = [], []

        for i, tc in enumerate(test_cases):
            question = tc["question"]
            ref_answer = tc.get("answer", "")
            relevant_ids = tc.get("relevant_ids", [])

            start = time.time()
            response = pipeline.query(question)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            # ROUGE
            if ref_answer:
                rouge = self.rouge_scores(response.answer, ref_answer)
                all_rouge1.append(rouge["rouge1"])
                all_rouge2.append(rouge["rouge2"])
                all_rougeL.append(rouge["rougeL"])
                predictions.append(response.answer)
                references.append(ref_answer)

            # Retrieval metrics
            if relevant_ids:
                retrieved_ids = [s["source"] for s in response.sources]
                all_precision.append(self.precision_at_k(retrieved_ids, relevant_ids))
                all_recall.append(self.recall_at_k(retrieved_ids, relevant_ids))
                all_mrr.append(self.mean_reciprocal_rank(retrieved_ids, relevant_ids))
                all_ndcg.append(self.ndcg_at_k(retrieved_ids, relevant_ids))

            if verbose and (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i + 1}/{len(test_cases)} samples")

        results = {
            "num_samples": len(test_cases),
            "avg_latency_ms": round(np.mean(latencies), 2),
            "p95_latency_ms": round(np.percentile(latencies, 95), 2),
        }

        if all_rouge1:
            results.update({
                "rouge1": round(np.mean(all_rouge1), 4),
                "rouge2": round(np.mean(all_rouge2), 4),
                "rougeL": round(np.mean(all_rougeL), 4),
            })

        if all_precision:
            results.update({
                "precision@5": round(np.mean(all_precision), 4),
                "recall@5":    round(np.mean(all_recall), 4),
                "mrr":         round(np.mean(all_mrr), 4),
                "ndcg@5":      round(np.mean(all_ndcg), 4),
            })

        if self.use_bertscore and predictions:
            results.update(self.bertscore(predictions, references))

        return results
