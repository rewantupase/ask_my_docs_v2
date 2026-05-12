"""
src/pipeline.py — Full RAG pipeline.

This module wires all components into a single callable.
It is the entry point for answering questions.

Pipeline stages:
  1. Hybrid retrieve   : BM25 + vector search → top 20 candidates (recall)
  2. Cross-encoder     : rerank top 20 → top 5 (precision)
  3. Build context     : format chunks with [Chunk N] IDs for citation
  4. Generate answer   : LLM reads context + question → raw answer
  5. Parse citations   : extract and validate [Chunk N] references
  6. Enforce citations : block answers without valid citations

This separation of concerns means each stage can be swapped independently:
  - Swap ChromaDB for Weaviate: change vectorstore.py, nothing else changes
  - Swap Llama for Mistral: change OLLAMA_MODEL in config.py
  - Swap MiniLM for a larger model: change EMBEDDING_MODEL in config.py
  - Add query expansion: insert before stage 1
  - Add re-retrieval (if answer is bad): add a feedback loop after stage 5
"""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from src.vectorstore  import get_vectorstore, get_all_chunks
from src.bm25_index   import get_bm25_index
from src.retriever    import HybridRetriever
from src.reranker     import rerank
from src.citations    import build_cited_context, parse_and_validate, RAGResponse
from src.generator    import generate_answer

logger = logging.getLogger(__name__)


def build_pipeline():
    """
    Initialise all pipeline components.
    Call this once at startup — it loads the embedding model, reranker,
    ChromaDB, and BM25 index into memory.

    Returns a callable: query_str → RAGResponse
    """
    logger.info("Building RAG pipeline...")

    # Load vector store (ChromaDB, SQLite-backed)
    vectorstore = get_vectorstore()

    # Load all chunks from ChromaDB to build (or restore) BM25 index
    all_chunks = get_all_chunks()
    if not all_chunks:
        raise RuntimeError(
            "No documents indexed yet.\n"
            "Run: python scripts/ingest.py --source <file_or_url>"
        )

    # Build (or load from disk) the BM25 index
    bm25_index = get_bm25_index(chunks=all_chunks)

    # Hybrid retriever: fuses BM25 + vector via RRF
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        bm25_index=bm25_index,
    )

    logger.info(f"Pipeline ready. {len(all_chunks)} chunks indexed.")

    def ask(question: str) -> RAGResponse:
        """
        Full RAG pipeline: question → cited answer.

        Args:
            question: natural language question string

        Returns:
            RAGResponse with answer, citations, and quality signals
        """
        logger.info(f"\n{'─'*50}")
        logger.info(f"Question: {question}")

        # ── Stage 1: Hybrid retrieval (recall) ────────────────────────────────
        candidates = retriever.retrieve(
            query=question,
            top_k=config.VECTOR_TOP_K,  # retrieve top 20 candidates
        )
        logger.info(f"Stage 1 — Hybrid retrieval: {len(candidates)} candidates")

        if not candidates:
            return RAGResponse(
                question=question,
                answer=config.CANNOT_ANSWER_PHRASE,
                citations=[],
                cannot_answer=True,
            )

        # ── Stage 2: Cross-encoder reranking (precision) ──────────────────────
        reranked = rerank(
            query=question,
            candidates=candidates,
            top_k=config.RERANK_TOP_K,  # keep top 5 after reranking
        )
        logger.info(f"Stage 2 — Reranked: {len(reranked)} chunks selected")

        # ── Stage 3: Build cited context ──────────────────────────────────────
        context = build_cited_context(reranked)

        # ── Stage 4: Generate answer ──────────────────────────────────────────
        raw_answer = generate_answer(context=context, question=question)
        logger.info(f"Stage 4 — Generated answer ({len(raw_answer.split())} words)")

        # ── Stage 5: Parse + enforce citations ────────────────────────────────
        response = parse_and_validate(
            question=question,
            answer=raw_answer,
            chunks=reranked,
        )
        logger.info(
            f"Stage 5 — Citations: {response.citation_count} | "
            f"Blocked: {response.was_blocked} | "
            f"Cannot answer: {response.cannot_answer}"
        )

        return response

    return ask
