"""
Source-Aware Scorer:
Boosts retrieval scores for documents from trusted medical sources
(e.g., WHO, NIH, PubMed) to improve answer reliability.
"""

from typing import List, Dict


# Default trust tiers for medical sources
SOURCE_TRUST_TIERS = {
    "tier_1": {
        "sources": ["WHO", "NIH", "CDC", "FDA", "PubMed"],
        "multiplier": 1.30,
    },
    "tier_2": {
        "sources": ["Mayo Clinic", "WebMD", "Medscape", "UpToDate"],
        "multiplier": 1.15,
    },
    "tier_3": {
        "sources": ["MedQuAD"],
        "multiplier": 1.05,
    },
}


class SourceAwareScorer:
    """
    Applies source-based trust multipliers to retrieval scores.

    Highly trusted sources (WHO, NIH, PubMed) receive a score boost,
    ensuring the final answer is grounded in authoritative content.
    """

    def __init__(self, trusted_sources: List[str] = None, custom_tiers: Dict = None):
        self.tiers = custom_tiers or SOURCE_TRUST_TIERS
        # Build fast lookup: source_name → multiplier
        self._multiplier_map: Dict[str, float] = {}
        for tier_info in self.tiers.values():
            for src in tier_info["sources"]:
                self._multiplier_map[src.lower()] = tier_info["multiplier"]

        # Additional user-specified trusted sources (tier 1)
        if trusted_sources:
            for src in trusted_sources:
                if src.lower() not in self._multiplier_map:
                    self._multiplier_map[src.lower()] = SOURCE_TRUST_TIERS["tier_1"]["multiplier"]

    def boost(self, candidates: List[Dict]) -> List[Dict]:
        """
        Apply trust multipliers to each candidate's score.

        Args:
            candidates: Chunk dicts with 'score' and 'source' keys.

        Returns:
            Candidates with boosted scores, re-sorted by score descending.
        """
        for doc in candidates:
            source = doc.get("source", "").lower()
            multiplier = 1.0
            # Match on substring (e.g., "pubmed_article_123" → pubmed)
            for known_src, mult in self._multiplier_map.items():
                if known_src in source:
                    multiplier = mult
                    break
            doc["score"] = doc.get("score", 0.0) * multiplier
            doc["trust_multiplier"] = multiplier

        return sorted(candidates, key=lambda d: d["score"], reverse=True)

    def get_trust_level(self, source: str) -> str:
        """Return the trust tier label for a source name."""
        source_lower = source.lower()
        for tier_name, tier_info in self.tiers.items():
            for src in tier_info["sources"]:
                if src.lower() in source_lower:
                    return tier_name
        return "unranked"
