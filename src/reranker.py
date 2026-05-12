"""
src/reranker.py — Cross-encoder reranking.

Uses _force_cpu_alloc() from embeddings.py to safely load CrossEncoder
without hitting the meta-tensor error.
"""

import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.embeddings import _force_cpu_alloc

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_reranker_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    with _reranker_lock:
        logger.info(f"Loading reranker: {config.RERANKER_MODEL}")

        with _force_cpu_alloc():
            reranker = CrossEncoder(
                config.RERANKER_MODEL,
                max_length=512,
                device="cpu",
            )

        # Materialize any remaining meta params
        for name, param in list(reranker.model.named_parameters()):
            if param.is_meta:
                logger.warning(f"Materializing meta param: {name}")
                param.data = torch.empty(param.shape, dtype=param.dtype, device="cpu")

        # Move to target device if not CPU
        if config.EMBEDDING_DEVICE != "cpu":
            reranker.model = reranker.model.to(config.EMBEDDING_DEVICE)

        logger.info("Reranker loaded.")
        return reranker


def rerank(
    query: str,
    candidates: List[Document],
    top_k: int = None,
) -> List[Tuple[Document, float]]:
    top_k = top_k or config.RERANK_TOP_K

    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    top = ranked[:top_k]
    logger.info(
        f"Reranked {len(candidates)} → top {len(top)} "
        f"(top score: {top[0][1]:.3f}, bottom score: {top[-1][1]:.3f})"
    )
    return top