"""
tests/test_pipeline.py — Unit tests for Ask My Docs.

Run with: python -m pytest tests/ -v

These tests use mock/minimal fixtures so they don't require
Ollama or the full model stack to be running.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Test: Chunking ────────────────────────────────────────────────────────────

class TestChunking:
    """Tests for the document chunking logic."""

    def test_chunk_size_within_bounds(self):
        """Chunks should be close to the configured size, never wildly over."""
        from langchain_core.documents import Document
        from src.ingestion import chunk_documents
        import config

        # Create a large fake document
        long_text = "This is a sentence about machine learning. " * 200
        docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]

        chunks = chunk_documents(docs)

        assert len(chunks) > 1, "Long document should produce multiple chunks"
        for chunk in chunks:
            # Each chunk should be ≤ chunk_size + overlap (with some margin)
            assert len(chunk.page_content) <= config.CHUNK_SIZE * 2, (
                f"Chunk too large: {len(chunk.page_content)} chars"
            )

    def test_chunk_metadata_injected(self):
        """Every chunk must have chunk_id, chunk_index, source, preview."""
        from langchain_core.documents import Document
        from src.ingestion import chunk_documents

        docs = [Document(
            page_content="Test content. " * 50,
            metadata={"source": "test_paper.pdf", "page": 3}
        )]
        chunks = chunk_documents(docs)

        for chunk in chunks:
            assert "chunk_id"    in chunk.metadata, "chunk_id missing"
            assert "chunk_index" in chunk.metadata, "chunk_index missing"
            assert "source"      in chunk.metadata, "source missing"
            assert "preview"     in chunk.metadata, "preview missing"

    def test_chunk_ids_are_unique(self):
        """Each chunk must have a globally unique ID."""
        from langchain_core.documents import Document
        from src.ingestion import chunk_documents

        docs = [Document(
            page_content="Different content for chunk " + str(i) + ". " * 20,
            metadata={"source": "test.pdf"}
        ) for i in range(5)]

        chunks = chunk_documents(docs)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected!"

    def test_overlap_produces_shared_content(self):
        """Adjacent chunks should share some content (the overlap)."""
        from langchain_core.documents import Document
        from src.ingestion import chunk_documents

        # Repeated sentences so we can detect overlap
        sentences = [f"Sentence number {i} about deep learning. " for i in range(100)]
        text = "".join(sentences)
        docs = [Document(page_content=text, metadata={"source": "overlap_test.txt"})]

        chunks = chunk_documents(docs)
        if len(chunks) >= 2:
            # The end of chunk 0 should appear in the start of chunk 1
            end_of_first   = chunks[0].page_content[-200:]
            start_of_second = chunks[1].page_content[:200]
            # At least some tokens should overlap
            words_end   = set(end_of_first.split())
            words_start = set(start_of_second.split())
            overlap_words = words_end & words_start
            assert len(overlap_words) > 0, "Expected overlap between adjacent chunks"


# ── Test: BM25 ────────────────────────────────────────────────────────────────

class TestBM25:
    """Tests for the BM25 keyword index."""

    def _make_chunks(self):
        from langchain_core.documents import Document
        return [
            Document(
                page_content="The Transformer uses scaled dot-product attention mechanism.",
                metadata={"chunk_id": "c0", "chunk_index": 0, "source": "paper.pdf"}
            ),
            Document(
                page_content="Adam optimizer with learning rate warmup schedule.",
                metadata={"chunk_id": "c1", "chunk_index": 1, "source": "paper.pdf"}
            ),
            Document(
                page_content="BLEU score evaluation on WMT 2014 English-German translation.",
                metadata={"chunk_id": "c2", "chunk_index": 2, "source": "paper.pdf"}
            ),
        ]

    def test_bm25_returns_results(self):
        from src.bm25_index import BM25Index
        index = BM25Index()
        index.build(self._make_chunks())
        results = index.search("attention mechanism", top_k=3)
        assert len(results) > 0, "BM25 returned no results"

    def test_bm25_exact_match_ranks_first(self):
        """A query with exact keywords should rank the matching chunk first."""
        from src.bm25_index import BM25Index
        index = BM25Index()
        chunks = self._make_chunks()
        index.build(chunks)

        results = index.search("BLEU score WMT English German", top_k=3)
        assert results, "No results for BLEU query"
        top_doc, top_score = results[0]
        assert "BLEU" in top_doc.page_content, (
            f"Expected BLEU chunk at rank 1, got: {top_doc.page_content[:60]}"
        )

    def test_bm25_save_load_roundtrip(self, tmp_path):
        """Save and load the BM25 index; search results must be consistent."""
        from src.bm25_index import BM25Index
        index = BM25Index()
        index.build(self._make_chunks())

        save_path = tmp_path / "bm25_test.pkl"
        index.save(save_path)

        loaded = BM25Index()
        assert loaded.load(save_path), "Failed to load BM25 index from disk"
        results = loaded.search("attention", top_k=2)
        assert len(results) > 0, "Loaded BM25 index returned no results"


# ── Test: RRF fusion ──────────────────────────────────────────────────────────

class TestRRFFusion:
    """Tests for the Reciprocal Rank Fusion logic."""

    def test_rrf_score_decreases_with_rank(self):
        """Higher rank (worse position) should give lower RRF score."""
        from src.retriever import _rrf_score
        scores = [_rrf_score(r) for r in range(10)]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i+1], (
                f"RRF score should decrease with rank: rank {i} score {scores[i]:.5f} "
                f"not > rank {i+1} score {scores[i+1]:.5f}"
            )

    def test_rrf_double_rank_beats_single(self):
        """A chunk ranked #1 in both systems should score higher than #1 in one."""
        from src.retriever import _rrf_score
        # Ranked 1st in both vector and BM25
        double_top = _rrf_score(0) + _rrf_score(0)
        # Ranked 1st in only vector, unranked in BM25
        single_top = _rrf_score(0)
        assert double_top > single_top


# ── Test: Citation enforcement ────────────────────────────────────────────────

class TestCitationEnforcement:
    """Tests for citation parsing and the no-citation guard."""

    def _make_chunks(self):
        from langchain_core.documents import Document
        chunks_with_scores = [
            (Document(
                page_content="The Transformer uses Adam optimizer.",
                metadata={"chunk_id": "c0", "source": "paper.pdf", "page": 5,
                          "preview": "The Transformer uses Adam optimizer."}
            ), 0.92),
            (Document(
                page_content="BLEU score of 28.4 on WMT14 EN-DE.",
                metadata={"chunk_id": "c1", "source": "paper.pdf", "page": 9,
                          "preview": "BLEU score of 28.4 on WMT14 EN-DE."}
            ), 0.85),
        ]
        return chunks_with_scores

    def test_valid_citation_parsed(self):
        """[Chunk 1] in the answer should be resolved to a Citation object."""
        from src.citations import parse_and_validate
        chunks = self._make_chunks()
        answer = "The model uses Adam optimizer [Chunk 1] with a warmup schedule."
        response = parse_and_validate("What optimizer?", answer, chunks)
        assert not response.was_blocked
        assert not response.cannot_answer
        assert response.citation_count == 1
        assert response.citations[0].chunk_number == 1

    def test_no_citation_blocked(self):
        """An answer without any [Chunk N] reference should be blocked."""
        from src.citations import parse_and_validate
        chunks = self._make_chunks()
        answer = "The model uses Adam optimizer with warmup."  # no citation!
        response = parse_and_validate("What optimizer?", answer, chunks)
        assert response.was_blocked, "Expected answer without citation to be blocked"

    def test_cannot_answer_phrase_accepted(self):
        """The model saying 'I cannot answer' should be accepted (not blocked)."""
        from src.citations import parse_and_validate
        import config
        chunks = self._make_chunks()
        response = parse_and_validate(
            "What is the square root of a transformer?",
            config.CANNOT_ANSWER_PHRASE,
            chunks,
        )
        assert response.cannot_answer
        assert not response.was_blocked

    def test_out_of_range_citation_ignored(self):
        """[Chunk 99] when only 2 chunks exist should be ignored silently."""
        from src.citations import parse_and_validate
        chunks = self._make_chunks()
        answer = "Some claim [Chunk 99] about something."
        response = parse_and_validate("Test?", answer, chunks)
        # [Chunk 99] is invalid → no valid citations → should be blocked
        assert response.was_blocked or response.citation_count == 0

    def test_multiple_citations_extracted(self):
        """Answer citing [Chunk 1] and [Chunk 2] should produce 2 citations."""
        from src.citations import parse_and_validate
        chunks = self._make_chunks()
        answer = "Uses Adam [Chunk 1] and achieves 28.4 BLEU [Chunk 2]."
        response = parse_and_validate("Performance?", answer, chunks)
        assert response.citation_count == 2
        numbers = {c.chunk_number for c in response.citations}
        assert numbers == {1, 2}


# ── Test: Context builder ──────────────────────────────────────────────────────

class TestContextBuilder:
    def test_context_includes_chunk_numbers(self):
        """Built context must include [Chunk N] headers."""
        from langchain_core.documents import Document
        from src.citations import build_cited_context

        chunks = [
            (Document(page_content="First chunk content.", metadata={"source": "a.pdf", "page": 1}), 0.9),
            (Document(page_content="Second chunk content.", metadata={"source": "a.pdf", "page": 2}), 0.8),
        ]
        context = build_cited_context(chunks)
        assert "[Chunk 1]" in context
        assert "[Chunk 2]" in context
        assert "First chunk content." in context
        assert "Second chunk content." in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
