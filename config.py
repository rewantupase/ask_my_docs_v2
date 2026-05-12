"""
config.py — Central configuration for Ask My Docs.

Every tunable parameter lives here. Change here, affects everywhere.
No magic numbers scattered across the codebase.
"""

import os
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
CHROMA_DIR     = BASE_DIR / "chroma_db"
DOCS_DIR       = BASE_DIR / "docs"
PROMPTS_DIR    = BASE_DIR / "prompts"
EVAL_DIR       = BASE_DIR / "eval"
BM25_INDEX_DIR = BASE_DIR / "chroma_db" / "bm25_index.pkl"  # persisted BM25

# ── Embedding model ───────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, ~80MB, runs on CPU, excellent quality
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE  = "cpu"        # change to "cuda" if you have a GPU
EMBEDDING_DIM     = 384

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 600          # target tokens per chunk (500-800 sweet spot)
CHUNK_OVERLAP     = 100          # overlap prevents boundary context loss
CHUNK_SEPARATORS  = ["\n\n", "\n", ". ", " ", ""]  # try paragraph → sentence → word

# ── Retrieval ─────────────────────────────────────────────────────────────────
VECTOR_TOP_K      = 20           # how many candidates vector search returns
BM25_TOP_K        = 20           # how many candidates BM25 returns
RRF_K_CONSTANT    = 60           # RRF fusion constant (60 from TREC paper)
RERANK_TOP_K      = 5            # how many chunks survive reranking → LLM context

# ── Reranker ──────────────────────────────────────────────────────────────────
# ms-marco-MiniLM-L-6-v2: trained on 500K MS MARCO passage pairs, ~70MB
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── LLM (local via Ollama) ────────────────────────────────────────────────────
OLLAMA_MODEL      = "mistral"    # run: ollama pull mistral  (Mistral 7B v0.3)
OLLAMA_BASE_URL   = "http://localhost:11434"
LLM_TEMPERATURE   = 0.1          # low = more factual, less creative
LLM_MAX_TOKENS    = 1024

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION_NAME   = "ask_my_docs"

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPT_VERSION    = "v1"         # which prompts/vX.yaml to load

# ── Evaluation thresholds (CI/CD gate) ────────────────────────────────────────
MIN_FAITHFULNESS        = 0.85   # answer must be grounded in retrieved chunks
MIN_ANSWER_RELEVANCY    = 0.80   # answer must address the question
MIN_CONTEXT_PRECISION   = 0.75   # retrieved chunks must be relevant

# ── Citation enforcement ──────────────────────────────────────────────────────
# If answer has no citations AND does not say "cannot answer" → block it
REQUIRE_CITATIONS       = True
CANNOT_ANSWER_PHRASE    = "I cannot answer this question from the provided documents."
