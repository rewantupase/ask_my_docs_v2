"""
src/retriever.py — Hybrid retrieval using Reciprocal Rank Fusion (RRF).

This is the heart of Phase 2. It combines two fundamentally different
retrieval signals into a single ranked list.

The two signals:
  1. Vector search (semantic)  — finds chunks that MEAN the same as the query
  2. BM25 (keyword/sparse)     — finds chunks that CONTAIN the same words

Why not just pick one?
  A pure vector system fails on: "What is the BLEU score of T5-large on WMT14?"
    → The model might return chunks about BLEU in general (semantically close)
      rather than the chunk with the exact T5-large WMT14 results.
    → BM25 would find "T5-large" and "WMT14" as exact token matches.

  A pure BM25 system fails on: "How does the model avoid overfitting?"
    → The paper might say "regularization prevents generalization collapse"
    → BM25 misses this because "overfitting" ≠ "generalization collapse"
    → Vector search finds it because they're semantically equivalent.

Reciprocal Rank Fusion (Cormack et al., 2009):
  Each system ranks chunks 1..N.
  RRF score = Σ 1/(rank + k)   where k=60 (empirically tuned constant)

  Why 1/(rank + k)?
    - Rank 1 from vector + Rank 1 from BM25 → very high fused score
    - Rank 1 from vector + Rank 100 from BM25 → moderate fused score
    - The +k constant dampens the difference between ranks 1-5
      (all near the top matter; don't over-reward being #1 vs #2)
    - k=60 was determined by Cormack et al. via systematic experiments
      on TREC data — it generalizes well across domains

  Key property: RRF is score-agnostic. It doesn't care that BM25 scores
  are in [0, ∞) and cosine similarities are in [0, 1]. It only uses rank
  positions, so no normalization is needed.
"""

import logging
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines ChromaDB vector search and BM25 keyword search
    via Reciprocal Rank Fusion.
    """

    def __init__(self, vectorstore, bm25_index):
        self.vectorstore = vectorstore
        self.bm25_index  = bm25_index

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Full hybrid retrieval pipeline:
          1. Vector search → top VECTOR_TOP_K candidates
          2. BM25 search   → top BM25_TOP_K candidates
          3. RRF fusion    → unified ranking
          4. Return top_k documents

        Returns deduplicated Documents sorted by fused relevance score.
        """
        top_k = top_k or (config.VECTOR_TOP_K + config.BM25_TOP_K) // 4

        # ── Step 1: Vector search ─────────────────────────────────────────────
        vec_results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=config.VECTOR_TOP_K
        )
        # vec_results: List[(Document, float)]  score ∈ [0, 1] higher=better

        # ── Step 2: BM25 search ───────────────────────────────────────────────
        bm25_results = self.bm25_index.search(query, top_k=config.BM25_TOP_K)
        # bm25_results: List[(Document, float)]  score ∈ [0, ∞) higher=better

        # ── Step 3: RRF fusion ────────────────────────────────────────────────
        fused_scores: dict[str, float] = {}   # chunk_id → fused score
        chunk_lookup: dict[str, Document] = {}

        # Process vector results
        for rank, (doc, _score) in enumerate(vec_results):
            cid = doc.metadata.get("chunk_id", doc.page_content[:50])
            chunk_lookup[cid] = doc
            fused_scores[cid] = fused_scores.get(cid, 0.0) + _rrf_score(rank)

        # Process BM25 results
        for rank, (doc, _score) in enumerate(bm25_results):
            cid = doc.metadata.get("chunk_id", doc.page_content[:50])
            chunk_lookup[cid] = doc
            fused_scores[cid] = fused_scores.get(cid, 0.0) + _rrf_score(rank)

        # ── Step 4: Sort by fused score, return top_k ─────────────────────────
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_chunks = [chunk_lookup[cid] for cid, _ in ranked[:top_k]]

        logger.info(
            f"Hybrid retrieval: {len(vec_results)} vec + {len(bm25_results)} bm25 "
            f"→ {len(fused_scores)} unique → returning top {len(top_chunks)}"
        )
        return top_chunks


def _rrf_score(rank: int, k: int = None) -> float:
    """
    Reciprocal Rank Fusion score for a document at the given rank (0-indexed).
    Score = 1 / (rank + 1 + k)  where k = config.RRF_K_CONSTANT (default 60)
    
    Example scores:
      rank 0  (1st place)  → 1/61  ≈ 0.0164
      rank 4  (5th place)  → 1/65  ≈ 0.0154
      rank 19 (20th place) → 1/80  ≈ 0.0125
    
    The difference between rank 1 and rank 5 is small (0.0164 vs 0.0154),
    so a document that ranks well in BOTH systems easily outscores one that
    ranks #1 in only one.
    """
    k = k or config.RRF_K_CONSTANT
    return 1.0 / (rank + 1 + k)
