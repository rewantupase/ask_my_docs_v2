"""
src/bm25_index.py — BM25 keyword search index.

BM25 (Best Match 25) is the gold standard keyword retrieval algorithm.
It is used by Elasticsearch, Solr, and Lucene under the hood.

Why BM25 alongside vector search?
  Vector search understands MEANING but struggles with:
    - Rare terms: "RLHF", "PPO", "GPT-4o" → all land in similar regions
    - Exact phrases: "p < 0.001" or "ablation study on layer 12"
    - Model version numbers: "BERT-large-uncased" vs "BERT-base-cased"
    - Typos and OOV tokens: vector for "transfomer" ≠ "transformer"

  BM25 matches EXACT TOKENS and handles these cases perfectly.
  Combining both covers what neither can alone.

How BM25 works:
  Score(doc, query) = Σ IDF(t) × TF(t, doc) × (k1+1) / (TF(t,doc) + k1 × (1-b+b×|doc|/avgdl))

  Where:
    TF   = term frequency in doc (how often the query word appears)
    IDF  = inverse document frequency (rare words score higher)
    k1   = 1.5 (term frequency saturation — beyond a point, more occurrences
                 don't keep improving the score)
    b    = 0.75 (document length normalization — penalizes very long docs)
    avgdl = average document length in the corpus

  BM25Okapi is the variant used by rank-bm25 — "Okapi" refers to the
  Okapi IR system at City University London where it was developed.

Persistence:
  BM25 is re-built from the corpus text each time (it's fast — O(n) where
  n = total tokens). We persist chunk IDs so we can map BM25 rank → Document.
"""

import logging
import pickle
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BM25Index:
    """
    Wrapper around rank_bm25.BM25Okapi that:
      1. Tokenizes chunk text (simple whitespace + lowercase)
      2. Builds the BM25 index
      3. Supports top-K retrieval returning Documents (not just scores)
      4. Can be saved to disk and loaded back without re-ingesting
    """

    def __init__(self):
        self.bm25 = None
        self.chunks: List[Document] = []
        self._built = False

    def build(self, chunks: List[Document]) -> None:
        """
        Build the BM25 index from a list of chunk Documents.
        Tokenization: lowercase + whitespace split.
        
        For production, consider a proper tokenizer (NLTK word_tokenize)
        but whitespace works well for most technical/research text.
        """
        from rank_bm25 import BM25Okapi

        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunk list.")

        self.chunks = chunks
        tokenized_corpus = [
            self._tokenize(chunk.page_content) for chunk in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._built = True
        logger.info(f"BM25 index built over {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = None) -> List[tuple]:
        """
        Search the BM25 index.
        Returns list of (Document, bm25_score) tuples sorted by score descending.
        
        BM25 scores are corpus-relative and unbounded — do NOT compare raw
        BM25 scores against cosine similarities. Use RRF for fusion instead.
        """
        if not self._built:
            raise RuntimeError("BM25 index not built. Call build() first.")

        top_k = top_k or config.BM25_TOP_K
        import numpy as np

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:   # skip zero-score chunks (no query terms matched)
                results.append((self.chunks[idx], score))

        return results

    def save(self, path: Path = None) -> None:
        """Persist the BM25 index to disk so it survives process restarts."""
        path = path or config.BM25_INDEX_DIR
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        logger.info(f"BM25 index saved to {path}")

    def load(self, path: Path = None) -> bool:
        """
        Load a previously saved BM25 index.
        Returns True if loaded successfully, False if file not found.
        """
        path = path or config.BM25_INDEX_DIR
        if not Path(path).exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        self._built = True
        logger.info(f"BM25 index loaded from {path} — {len(self.chunks)} chunks")
        return True

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple tokenizer: lowercase + whitespace split.
        Strips punctuation from token edges (e.g. "word." → "word").
        """
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9_\-]+\b', text)
        return tokens


# Module-level singleton — build once per process
_bm25_index: BM25Index = None


def get_bm25_index(chunks: List[Document] = None, force_rebuild: bool = False) -> BM25Index:
    """
    Get (or build) the BM25 index singleton.
    
    Call sequence:
      1. If index already in memory → return it
      2. Else if saved to disk → load it
      3. Else build from chunks (must be provided)
    
    After building, always saves to disk for next session.
    """
    global _bm25_index

    if _bm25_index is not None and not force_rebuild:
        return _bm25_index

    _bm25_index = BM25Index()

    # Try loading from disk first
    if not force_rebuild and _bm25_index.load():
        return _bm25_index

    # Build from chunks
    if chunks is None:
        raise ValueError(
            "BM25 index not found on disk. Provide chunks to build it, "
            "or run: python scripts/ingest.py first."
        )

    _bm25_index.build(chunks)
    _bm25_index.save()
    return _bm25_index
