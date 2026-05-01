#!/usr/bin/env python3
"""
main.py — CLI entry point for the Hybrid RAG Medical QA system.

Usage:
  python main.py --mode index   --config configs/config.yaml
  python main.py --mode query   --config configs/config.yaml --question "What are symptoms of diabetes?"
  python main.py --mode eval    --config configs/config.yaml --eval-data data/test_cases.json
  python main.py --mode demo    --config configs/config.yaml
"""

import argparse
import json
import yaml
import logging
from pathlib import Path

from src.pipeline import MedicalRAGPipeline
from src.data_loaders import MedQuADLoader, PubMedLoader, WHOLoader
from src.evaluator import RAGEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_documents(config: dict) -> list:
    """Load documents from all configured data sources."""
    documents = []
    data_cfg = config.get("data", {})

    medquad_path = data_cfg.get("medquad_csv")
    if medquad_path and Path(medquad_path).exists():
        loader = MedQuADLoader()
        documents.extend(loader.load(medquad_path))
    else:
        logger.warning(f"MedQuAD CSV not found: {medquad_path}")

    pubmed_path = data_cfg.get("pubmed_jsonl")
    if pubmed_path and Path(pubmed_path).exists():
        loader = PubMedLoader()
        documents.extend(loader.load(pubmed_path, max_docs=10000))
    else:
        logger.warning(f"PubMed JSONL not found: {pubmed_path}")

    who_dir = data_cfg.get("who_dir")
    if who_dir and Path(who_dir).exists():
        loader = WHOLoader()
        documents.extend(loader.load_directory(who_dir))
    else:
        logger.warning(f"WHO directory not found: {who_dir}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def run_index(pipeline: MedicalRAGPipeline, config: dict) -> None:
    """Index all documents."""
    documents = load_documents(config)
    if not documents:
        logger.error("No documents loaded. Please check your data paths in config.yaml.")
        return
    pipeline.index_documents(documents)
    logger.info("Indexing complete.")


def run_query(pipeline: MedicalRAGPipeline, question: str) -> None:
    """Run a single query and print the result."""
    print(f"\n{'='*60}")
    print(f"  Question: {question}")
    print(f"{'='*60}\n")

    response = pipeline.query(question, verbose=True)

    print(f"Answer:\n{response.answer}\n")
    print(f"Confidence : {response.confidence:.2%}")
    print(f"Latency    : {response.latency_ms:.1f} ms\n")
    print("Sources:")
    for i, src in enumerate(response.sources, 1):
        score_str = f"{src['score']:.4f}"
        print(f"  [{i}] {src['source']}  (score: {score_str})")
        print(f"      {src['text'][:120]}...")
    print()


def run_eval(pipeline: MedicalRAGPipeline, eval_data_path: str, config: dict) -> None:
    """Run full evaluation on a test set."""
    with open(eval_data_path, "r") as f:
        test_cases = json.load(f)

    evaluator = RAGEvaluator(use_bertscore=config.get("use_bertscore", False))
    results = evaluator.evaluate_dataset(pipeline, test_cases, verbose=True)

    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)
    for metric, value in results.items():
        print(f"  {metric:<25} {value}")
    print("="*50 + "\n")


def run_demo(pipeline: MedicalRAGPipeline) -> None:
    """Interactive demo mode."""
    print("\n" + "="*60)
    print("  Hybrid RAG Medical QA — Interactive Demo")
    print("  Type 'quit' to exit")
    print("="*60 + "\n")

    while True:
        try:
            question = input("Ask a medical question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        run_query(pipeline, question)


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG Medical QA System")
    parser.add_argument("--mode", choices=["index", "query", "eval", "demo"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--question", type=str, help="Question for query mode")
    parser.add_argument("--eval-data", type=str, help="Path to test cases JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = MedicalRAGPipeline(config)

    if args.mode == "index":
        run_index(pipeline, config)

    elif args.mode == "query":
        if not args.question:
            parser.error("--question is required for query mode")
        run_index(pipeline, config)
        run_query(pipeline, args.question)

    elif args.mode == "eval":
        if not args.eval_data:
            parser.error("--eval-data is required for eval mode")
        run_index(pipeline, config)
        run_eval(pipeline, args.eval_data, config)

    elif args.mode == "demo":
        run_index(pipeline, config)
        run_demo(pipeline)


if __name__ == "__main__":
    main()
