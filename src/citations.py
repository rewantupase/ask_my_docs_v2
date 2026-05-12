"""
src/citations.py — Citation enforcement and structured output.

This module does two things:
  1. Build the cited context string the LLM sees (chunk texts with IDs)
  2. Parse and validate the LLM's answer (did it actually cite? are the IDs valid?)
  3. Block answers that claim to know without evidence

Why is citation enforcement "huge"?
  Without it:
    - LLM sees chunks about transformers
    - LLM "knows" from training that BERT uses Adam
    - LLM says "BERT uses Adam" even if no retrieved chunk mentions it
    - User trusts it → hallucination passes undetected

  With enforcement:
    - Prompt instructs: every claim must have [Chunk N]
    - Parser checks: did the model actually cite?
    - Gate: if no citation and no "cannot answer" → block the response
    - User sees which exact chunk supports which claim → auditable

Citation format used: [Chunk N]
  Simple, unambiguous, easy to parse with regex.
  The LLM is instructed to use this exact format.
  We count occurrences and validate N ≤ number of retrieved chunks.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A single citation extracted from the LLM answer."""
    chunk_number: int          # as used in the answer: [Chunk 3] → 3
    chunk_index: int           # 0-based index into the context chunks list
    source: str                # original file path or URL
    page: Optional[int]        # page number (None for web/markdown)
    preview: str               # first 300 chars of the chunk text
    relevance_score: float     # cross-encoder score (higher = more relevant)


@dataclass
class RAGResponse:
    """
    Structured response from the RAG pipeline.
    Contains the answer, all validated citations, and quality signals.
    """
    question:        str
    answer:          str
    citations:       List[Citation]
    was_blocked:     bool  = False    # True if answer was blocked (no citation)
    cannot_answer:   bool  = False    # True if model said "I cannot answer"
    citation_count:  int  = 0
    sources_used:    List[str] = field(default_factory=list)

    def __post_init__(self):
        self.citation_count = len(self.citations)
        self.sources_used   = list({c.source for c in self.citations})

    def display(self) -> str:
        """Pretty-print for terminal output."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"QUESTION: {self.question}")
        lines.append(f"{'='*60}")

        if self.was_blocked:
            lines.append("\n⛔  Response blocked — model answered without citations.")
            lines.append(f"   Returning: {config.CANNOT_ANSWER_PHRASE}")
        elif self.cannot_answer:
            lines.append(f"\n📭  {config.CANNOT_ANSWER_PHRASE}")
        else:
            lines.append(f"\nANSWER:\n{self.answer}")
            lines.append(f"\n{'─'*60}")
            lines.append(f"CITATIONS ({self.citation_count}):")
            for i, cit in enumerate(self.citations, 1):
                lines.append(
                    f"\n  [{i}] Chunk {cit.chunk_number} "
                    f"| {cit.source} | Page {cit.page or 'N/A'}"
                )
                lines.append(f"       Score: {cit.relevance_score:.3f}")
                lines.append(f"       Preview: \"{cit.preview[:120]}...\"")

        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ── Context building ──────────────────────────────────────────────────────────

def build_cited_context(
    chunks: List[tuple],   # List of (Document, relevance_score)
) -> str:
    """
    Format retrieved chunks into the numbered context block the LLM receives.

    Each chunk is prefixed with [Chunk N] so the LLM can reference it.
    Source metadata is included so the LLM can see where the text came from
    (though citations come from our parser, not the LLM's metadata knowledge).

    Example output:
      [Chunk 1] Source: paper.pdf | Page: 3
      The transformer architecture uses scaled dot-product attention...

      ---

      [Chunk 2] Source: paper.pdf | Page: 7
      We train using Adam optimizer with β1=0.9, β2=0.98...
    """
    parts = []
    for i, (doc, score) in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "N/A")
        header = f"[Chunk {i}] Source: {source} | Page: {page}"
        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# ── Citation parsing and validation ───────────────────────────────────────────

_CITATION_PATTERN = re.compile(r'\[Chunk\s+(\d+)\]', re.IGNORECASE)


def parse_and_validate(
    question: str,
    answer: str,
    chunks: List[tuple],   # List of (Document, relevance_score)
) -> RAGResponse:
    """
    Parse the LLM answer for [Chunk N] citations, validate them, and
    enforce the "no citation → cannot answer" rule.

    Validation rules:
      1. Extract all [Chunk N] references from the answer
      2. Check each N is in range [1, len(chunks)]
      3. If no valid citations found AND answer doesn't say "cannot answer":
           → Block the answer (hallucination guard)
      4. If model said "cannot answer" → mark appropriately (correct behavior)
      5. Otherwise → build structured Citation objects
    """
    # Check if model refused to answer (correct behavior when no evidence)
    cannot_answer = config.CANNOT_ANSWER_PHRASE.lower() in answer.lower()

    if cannot_answer:
        return RAGResponse(
            question=question,
            answer=config.CANNOT_ANSWER_PHRASE,
            citations=[],
            cannot_answer=True,
        )

    # Extract citation numbers from answer
    raw_matches = _CITATION_PATTERN.findall(answer)   # e.g. ["1", "3", "1"]
    cited_numbers = sorted(set(int(m) for m in raw_matches))

    # Validate: filter out out-of-range citations
    valid_citations = []
    for num in cited_numbers:
        idx = num - 1   # convert 1-based display → 0-based list index
        if 0 <= idx < len(chunks):
            doc, score = chunks[idx]
            valid_citations.append(Citation(
                chunk_number=num,
                chunk_index=idx,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", None),
                preview=doc.metadata.get("preview", doc.page_content[:300]),
                relevance_score=float(score),
            ))
        else:
            logger.warning(f"Model cited [Chunk {num}] but only {len(chunks)} chunks exist.")

    # Enforcement gate
    if not valid_citations and config.REQUIRE_CITATIONS:
        logger.warning("Answer has no valid citations — blocking response.")
        return RAGResponse(
            question=question,
            answer=config.CANNOT_ANSWER_PHRASE,
            citations=[],
            was_blocked=True,
        )

    return RAGResponse(
        question=question,
        answer=answer,
        citations=valid_citations,
    )
