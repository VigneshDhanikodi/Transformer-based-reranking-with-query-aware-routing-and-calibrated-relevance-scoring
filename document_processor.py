"""
Document Processor: Parses PDF/HTML, cleans, and chunks documents
with configurable overlap for efficient retrieval.
"""

import re
import unicodedata
from typing import List, Dict, Optional
from pathlib import Path


class DocumentProcessor:
    """
    Handles end-to-end document ingestion:
      1. Parse PDF / HTML / plain-text
      2. Clean and normalize text
      3. Split into overlapping chunks
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process(self, documents: List[Dict]) -> List[Dict]:
        """
        Process a list of raw documents into cleaned, chunked pieces.

        Args:
            documents: [{'text': str, 'source': str, 'metadata': dict}, ...]

        Returns:
            List of chunk dicts with text, source, chunk_id, and metadata.
        """
        all_chunks = []
        for doc in documents:
            text = self._clean(doc.get("text", ""))
            chunks = self._chunk(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "text": chunk,
                        "source": doc.get("source", "Unknown"),
                        "chunk_id": f"{doc.get('source', 'doc')}_{i}",
                        "metadata": doc.get("metadata", {}),
                    }
                )
        return all_chunks

    def process_file(self, filepath: str) -> List[Dict]:
        """
        Parse and process a single file (PDF, HTML, or .txt).
        Requires: PyMuPDF (fitz) for PDF, BeautifulSoup for HTML.
        """
        path = Path(filepath)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            text = self._parse_pdf(filepath)
        elif suffix in (".html", ".htm"):
            text = self._parse_html(filepath)
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        source_name = path.stem
        return self.process([{"text": text, "source": source_name, "metadata": {"file": filepath}}])

    # ------------------------------------------------------------------ #
    #  Parsing                                                             #
    # ------------------------------------------------------------------ #

    def _parse_pdf(self, filepath: str) -> str:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(filepath)
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    def _parse_html(self, filepath: str) -> str:
        try:
            from bs4 import BeautifulSoup
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            # Remove scripts and styles
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator=" ")
        except ImportError:
            raise ImportError("BeautifulSoup not installed. Run: pip install beautifulsoup4")

    # ------------------------------------------------------------------ #
    #  Cleaning                                                            #
    # ------------------------------------------------------------------ #

    def _clean(self, text: str) -> str:
        """Normalize unicode, remove noise, collapse whitespace."""
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)        # remove non-ASCII
        text = re.sub(r"\s+", " ", text)                   # collapse whitespace
        text = re.sub(r"(\n\s*){3,}", "\n\n", text)       # max 2 consecutive newlines
        text = re.sub(r"[^\w\s.,;:!?()\-\/]", "", text)   # strip unusual punctuation
        return text.strip()

    # ------------------------------------------------------------------ #
    #  Chunking                                                            #
    # ------------------------------------------------------------------ #

    def _chunk(self, text: str) -> List[str]:
        """
        Sliding-window word-level chunking with overlap.
        Preserves sentence boundaries where possible.
        """
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += self.chunk_size - self.chunk_overlap

        return [c for c in chunks if len(c.split()) >= 10]  # filter micro-chunks
