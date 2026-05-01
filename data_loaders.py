"""
Dataset Loaders:
Load and standardize MedQuAD, PubMed/PMC abstracts, and WHO guidelines
into the common document format for the RAG pipeline.
"""

import os
import csv
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Common document schema:
# {'text': str, 'source': str, 'metadata': {'title': str, 'url': str, ...}}


class MedQuADLoader:
    """
    Loads the MedQuAD dataset from a CSV file.
    Download from: https://github.com/abachaa/MedQuAD
                   https://www.kaggle.com/datasets/pythonapimaster/medquad-dataset
    """

    def load(self, csv_path: str) -> List[Dict]:
        """
        Args:
            csv_path: Path to medquad.csv (columns: qtype, Question, Answer, Source, ...)

        Returns:
            List of document dicts for the RAG pipeline.
        """
        documents = []
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"MedQuAD CSV not found: {csv_path}")

        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row.get("Question", "").strip()
                answer = row.get("Answer", "").strip()
                source = row.get("Source", "MedQuAD").strip()
                qtype = row.get("qtype", "").strip()

                if not question or not answer:
                    continue

                text = f"Q: {question}\nA: {answer}"
                documents.append({
                    "text": text,
                    "source": f"MedQuAD_{source}",
                    "metadata": {
                        "question": question,
                        "answer": answer,
                        "qtype": qtype,
                        "original_source": source,
                    },
                })

        logger.info(f"Loaded {len(documents)} MedQuAD QA pairs from {csv_path}")
        return documents

    def load_as_qa_pairs(self, csv_path: str) -> List[Dict]:
        """Load as (question, answer) pairs for evaluation."""
        qa_pairs = []
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("Question", "").strip()
                a = row.get("Answer", "").strip()
                if q and a:
                    qa_pairs.append({"question": q, "answer": a})
        return qa_pairs


class PubMedLoader:
    """
    Loads PubMed abstracts from a JSONL file.
    Each line: {"pmid": "...", "title": "...", "abstract": "...", "mesh_terms": [...]}

    To download PubMed abstracts:
    - PubMed FTP: https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
    - Or use the Entrez API: https://www.ncbi.nlm.nih.gov/books/NBK25499/
    """

    def load(self, jsonl_path: str, max_docs: Optional[int] = None) -> List[Dict]:
        """
        Args:
            jsonl_path: Path to pubmed_abstracts.jsonl
            max_docs:   Optional limit on number of documents to load.

        Returns:
            List of document dicts.
        """
        documents = []
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"PubMed JSONL not found: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_docs and len(documents) >= max_docs:
                    break
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                title = record.get("title", "").strip()
                abstract = record.get("abstract", "").strip()
                pmid = record.get("pmid", "")

                if not abstract:
                    continue

                text = f"{title}\n{abstract}" if title else abstract
                documents.append({
                    "text": text,
                    "source": f"PubMed_{pmid}",
                    "metadata": {
                        "title": title,
                        "pmid": pmid,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "mesh_terms": record.get("mesh_terms", []),
                    },
                })

        logger.info(f"Loaded {len(documents)} PubMed abstracts from {jsonl_path}")
        return documents


class WHOLoader:
    """
    Loads WHO guideline documents from a directory of plain-text or JSON files.
    Typically downloaded from: https://www.who.int/publications/
    """

    def load_directory(self, dir_path: str) -> List[Dict]:
        """
        Load all .txt and .json files in a directory as WHO documents.

        Args:
            dir_path: Directory containing WHO documents.

        Returns:
            List of document dicts.
        """
        documents = []
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"WHO data directory not found: {dir_path}")

        for file in path.iterdir():
            if file.suffix == ".txt":
                docs = self._load_txt(file)
            elif file.suffix == ".json":
                docs = self._load_json(file)
            else:
                continue
            documents.extend(docs)

        logger.info(f"Loaded {len(documents)} WHO documents from {dir_path}")
        return documents

    def _load_txt(self, path: Path) -> List[Dict]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if not text:
            return []
        return [{
            "text": text,
            "source": f"WHO_{path.stem}",
            "metadata": {"file": str(path), "title": path.stem},
        }]

    def _load_json(self, path: Path) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            return []

        documents = []
        for record in records:
            text = record.get("text") or record.get("content") or record.get("body", "")
            title = record.get("title", path.stem)
            if text:
                documents.append({
                    "text": text,
                    "source": f"WHO_{title.replace(' ', '_')}",
                    "metadata": {"title": title, "url": record.get("url", "")},
                })
        return documents
