<div align="center">

# Ask My Docs v2

### A Production-Grade RAG System — 100% Free, 100% Local, Full Web UI

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Mistral](https://img.shields.io/badge/LLM-Mistral%207B-FF7000?style=for-the-badge&logo=mistral&logoColor=white)](https://ollama.com/library/mistral)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6B35?style=for-the-badge)](https://www.trychroma.com/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Flask](https://img.shields.io/badge/Web%20UI-Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-15%20Passing-22C55E?style=for-the-badge&logo=pytest&logoColor=white)]()

<br/>

> **Ingest any PDF, Markdown file, or web page. Ask questions in plain English. Get cited, grounded answers.**
> No API keys. No cloud. No cost. Runs entirely on your machine.

</div>

---

## Table of Contents

- [What is This?](#what-is-this)
- [Why This Matters](#why-this-matters)
- [What's New in v2](#whats-new-in-v2)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Deep Dive](#pipeline-deep-dive)
  - [System Architecture](#system-architecture)
  - [Phase 1 — Document Ingestion](#phase-1--document-ingestion)
  - [Phase 2 — Hybrid Retrieval and Generation](#phase-2--hybrid-retrieval-and-generation)
  - [Phase 3 — Evaluation and CI/CD](#phase-3--evaluation-and-cicd)
- [Quickstart](#quickstart)
- [Web Interface](#web-interface)
- [REST API](#rest-api)
- [CLI Usage](#cli-usage)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Running Tests](#running-tests)
- [Key Design Decisions](#key-design-decisions)
- [Swapping Components](#swapping-components)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [References](#references)
- [License](#license)

---

## What is This?

**Ask My Docs** is a domain-specific question-answering system built on the RAG (Retrieval-Augmented Generation) architecture. Feed it your documents — research papers, legal contracts, technical manuals, internal wikis — and ask questions in plain English. Every answer comes with citations pointing to the exact source chunk that supports the claim.

**Without citations, answers are guesses. With citations, every claim is auditable.**

This is not a prototype. It implements the full production RAG stack:

- **Hybrid retrieval** — BM25 keyword search and vector similarity search fused via Reciprocal Rank Fusion
- **Cross-encoder reranking** — precision scoring after broad-net recall
- **Citation enforcement** — the system says *"I cannot answer"* rather than hallucinating
- **Versioned prompts** — stored in YAML config files, tracked by git like code
- **Offline RAGAS evaluation** — with a CI/CD quality gate on every pull request
- **Full web UI** — browser-based interface with drag-and-drop ingestion, streaming responses, and a document library

Everything runs locally and for free. No OpenAI key. No Pinecone account. No data sent anywhere.

---

## Why This Matters

Large language models hallucinate because they generate the most *probable* next token — not the most *grounded* one. RAG fixes this by forcing the model to answer only from retrieved evidence. Citations make every claim verifiable.

| Without RAG | With Ask My Docs |
|---|---|
| Model answers from training data | Model answers only from your documents |
| No way to verify a claim | Every claim cites its exact source chunk |
| Hallucination is invisible | Unsupported answers are blocked at the gate |
| Generic knowledge | Domain-specific precision |
| Stale information | Always reflects your latest documents |

---

## What's New in v2

v2 ships a full browser-based interface alongside the original CLI — no terminal required for day-to-day use.

| Feature | v1 | v2 |
|---|---|---|
| CLI ingestion and Q&A | Yes | Yes |
| Web UI with chat interface | No | Yes |
| Drag-and-drop file upload | No | Yes |
| Ingest URLs from the browser | No | Yes |
| Document library sidebar | No | Yes |
| Delete indexed documents | No | Yes |
| Dark mode | No | Yes |
| Streaming SSE responses | No | Yes |
| REST API (`/api/*`) | No | Yes |
| Flask web server (`app.py`) | No | Yes |

---

## Tech Stack

Every tool in this stack is free and open source. No API keys. No billing.

| Layer | Tool | Why |
|---|---|---|
| **Web UI** | Flask + vanilla JS | Single-file SPA, dark mode, SSE streaming, no build step |
| **Document Loading** | PyMuPDF, BeautifulSoup4, UnstructuredMarkdown | Layout-aware PDF parsing, clean web scraping, Markdown support |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Natural boundary splits: paragraph → sentence → word |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, ~80MB, CPU-friendly — within 5% of OpenAI on MTEB |
| **Vector Store** | ChromaDB (local SQLite) | Zero config, persists to disk, no server needed |
| **Keyword Search** | `rank-bm25` (BM25Okapi) | Exact phrase, model name, and rare term matching |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) | Score-agnostic rank merge from TREC research |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 70MB cross-encoder trained on 500K MS MARCO pairs |
| **LLM** | Mistral 7B v0.3 via Ollama | Best-in-class instruction following at 7B, ~4GB RAM |
| **Prompt Versioning** | YAML config files | Prompts are architecture — git-tracked like code |
| **Evaluation** | RAGAS (offline) | Faithfulness, relevancy, and context precision scoring |
| **Orchestration** | LangChain | Modular, swappable pipeline components |
| **CI/CD** | GitHub Actions | Auto-eval on every PR — fails build on quality regression |

---

## Project Structure

```
ask_my_docs/
│
├── app.py                         # Flask web server + REST API (v2)
├── config.py                      # Central config — every parameter in one place
├── requirements.txt               # All dependencies (pinned versions)
├── setup_check.py                 # Environment verification script
│
├── web/
│   └── index.html                 # Single-file SPA (HTML + CSS + JS, dark mode)
│
├── prompts/
│   └── v1.yaml                    # Versioned prompt templates (git-tracked)
│
├── src/                           # Core pipeline modules
│   ├── ingestion.py               # PDF / Markdown / Web URL → chunked Documents
│   ├── embeddings.py              # all-MiniLM-L6-v2 wrapper (singleton)
│   ├── vectorstore.py             # ChromaDB read / write / reset
│   ├── bm25_index.py              # BM25Okapi keyword index (persisted to disk)
│   ├── retriever.py               # Hybrid RRF fusion (BM25 + vector → top-20)
│   ├── reranker.py                # Cross-encoder reranker (top-20 → top-5)
│   ├── citations.py               # [Chunk N] parser + hallucination gate
│   ├── generator.py               # Ollama/Mistral call with prompt versioning
│   ├── pipeline.py                # Wires all stages: question → RAGResponse
│   └── __init__.py
│
├── scripts/                       # CLI entry points
│   ├── ingest.py                  # Ingest documents into the vector store
│   ├── ask.py                     # Ask questions (single or interactive mode)
│   └── evaluate.py                # Run RAGAS evaluation against golden dataset
│
├── eval/
│   └── golden_dataset.json        # Verified QA pairs for benchmarking
│
├── tests/
│   └── test_pipeline.py           # 15 unit tests (chunking, BM25, RRF, citations)
│
├── docs/                          # Drop your documents here
│
├── chroma_db/                     # Auto-created — vector store + BM25 index
│
└── .github/workflows/
    └── rag_quality_gate.yml       # CI/CD: auto-eval on every pull request
```

---

## Pipeline Deep Dive

### System Architecture

The diagram below shows the full end-to-end flow — from raw documents on the left, through hybrid retrieval and reranking, to a cited answer on the right. Refer back to this as you read through each phase.

<img src="workflow.png" alt="Ask My Docs — Full RAG Pipeline Architecture" width="100%"/>

---

### Phase 1 — Document Ingestion

```
PDF / Markdown / Web URL
         │
         ▼
   [ Document Loader ]        PyMuPDF (PDF) · UnstructuredMarkdown · WebBaseLoader
         │
         ▼
   [ Chunking ]               600 tokens · 100-token overlap · natural boundary splits
         │
         ▼
   [ Metadata Injection ]     chunk_id · source · page · preview (for citations)
         │
    ┌────┴────┐
    ▼         ▼
[ Embed ]  [ BM25 Index ]     all-MiniLM-L6-v2 (384-dim) · BM25Okapi keyword index
    │
    ▼
[ ChromaDB ]                  Local SQLite vector store — persisted to disk
```

**Why 600 tokens with 100-token overlap?**
Too small (below ~200 tokens) and chunks lose surrounding context. Too large (above ~1200 tokens) and retrieval becomes imprecise — the chunk contains too many ideas for a single query to match well. The 100-token overlap ensures sentences near chunk boundaries appear in both adjacent chunks so no context is lost at the seam.

**Why `RecursiveCharacterTextSplitter`?**
It tries natural boundaries first — paragraph, then line, then sentence, then word, then character. This keeps coherent ideas together instead of cutting mid-sentence at a hard character limit.

---

### Phase 2 — Hybrid Retrieval and Generation

```
User Question
      │
      ├──────────────────────┬─────────────────────┐
      ▼                      ▼                     │
[ Vector Search ]      [ BM25 Search ]             │
  semantic meaning       exact keywords            │
  top-20 results         top-20 results            │
      │                      │                     │
      └──────────┬───────────┘                     │
                 ▼                                 │
       [ RRF Fusion ]                              │
    Reciprocal Rank Fusion                         │
    score-agnostic rank merge                      │
                 │                                 │
                 ▼                                 │
    [ Cross-Encoder Reranker ]             [ Prompt Config ]
   ms-marco-MiniLM-L-6-v2                 prompts/v1.yaml
   top-20 → top-5 (precision)             versioned by git
                 │                                 │
                 └─────────────────┬───────────────┘
                                   ▼
                       [ Context Builder ]
                       [Chunk 1] source · page
                       [Chunk 2] source · page ...
                                   │
                                   ▼
                          [ Mistral 7B ]
                       via Ollama (local, free)
                                   │
                                   ▼
                       [ Citation Parser ]
                       extract [Chunk N] refs
                                   │
                                   ▼
                       [ Hallucination Gate ]
                       no citation?     → BLOCKED
                       cannot answer?   → ACCEPTED
                                   │
                                   ▼
                          [ RAGResponse ]
                       answer + citations + sources
```

**Why hybrid retrieval instead of vector search alone?**

| Signal | Wins on | Fails on |
|---|---|---|
| Vector search | Synonyms, paraphrases, concepts | Rare terms, model names, exact versions |
| BM25 | Exact phrases, acronyms, IDs | Semantic similarity, cross-lingual queries |
| Hybrid (RRF) | Both — typically 15–25% better than either alone | — |

**Why two retrieval stages — broad then precise?**
Bi-encoders are fast but encode query and document separately, so they miss fine-grained interactions. Cross-encoders read query and document together through self-attention, capturing nuances like negation and contrast — but are roughly 100x slower. The two-stage approach casts a wide net cheaply (top-20 via bi-encoder), then scores precisely (top-5 via cross-encoder).

**Why Reciprocal Rank Fusion?**
BM25 scores are unbounded; cosine similarities are bounded to [0, 1]. You cannot directly compare or sum them. RRF uses only rank positions (`score = 1 / (rank + 60)`), so no normalization is needed. The constant 60 was empirically determined from TREC experiments and generalizes well across retrieval tasks.

---

### Phase 3 — Evaluation and CI/CD

```
Golden Dataset (verified QA pairs)
              │
              ▼
  [ Run Pipeline on each QA pair ]
              │
              ▼
       [ RAGAS Scoring ]
  ┌──────────────────────────────────────┐
  │  faithfulness         >= 0.85        │  Is every claim grounded in context?
  │  answer_relevancy     >= 0.80        │  Does the answer address the question?
  │  context_precision    >= 0.75        │  Are the retrieved chunks relevant?
  └──────────────────────────────────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
  PASS (0)          FAIL (1)
  PR merges         PR blocked
```

**What is faithfulness?**
RAGAS decomposes every sentence in the generated answer into atomic claims, then checks whether each claim is supported by the retrieved chunks. The score equals supported claims divided by total claims. This is the automated hallucination detector — no human review required on every PR.

---

## Quickstart

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed
- Mistral 7B pulled locally

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ask_my_docs_v2.git
cd ask_my_docs_v2

# 2. Create and activate a virtual environment
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama and pull Mistral
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral            # Mistral 7B v0.3 — ~4GB, runs on CPU

# 5. Verify the environment
python setup_check.py
```

You should see `All checks passed!` before continuing.

### Launch

```bash
# Ingest a document
python scripts/ingest.py --source docs/your_document.pdf

# Start the web UI
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Web Interface

The v2 UI is a full single-page application served by Flask, with no build step or external frontend dependencies.

**Sidebar — document library**

- Lists all indexed documents with their chunk counts
- Upload new PDFs or Markdown files via drag-and-drop or the file picker
- Paste any URL to ingest a web page directly from the browser
- Delete any document from the index with one click — the BM25 index and pipeline singleton are rebuilt automatically

**Chat panel**

- Ask questions in natural language
- Answers stream in token-by-token via Server-Sent Events (SSE) — no waiting for the full response to complete
- Each answer displays expandable citation cards showing the source document, page number, and relevance score
- When no relevant context exists, the system responds with a clean "cannot answer" message rather than guessing
- Full dark mode support, respecting the system preference automatically

---

## REST API

The Flask server exposes a REST API you can call from any client or integrate into other tooling.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Health check — returns model names and total chunk count |
| `GET` | `/api/docs` | List all indexed documents with chunk counts |
| `POST` | `/api/ask` | Ask a question — SSE streaming response |
| `POST` | `/api/ingest/file` | Upload and ingest a PDF or Markdown file |
| `POST` | `/api/ingest/url` | Ingest a web page by URL |
| `DELETE` | `/api/docs/<name>` | Remove a document and all its chunks from the index |

**Ask a question**

```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What optimizer does the Transformer use?"}'
```

**Ingest a URL**

```bash
curl -X POST http://localhost:5000/api/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/abs/1706.03762"}'
```

**Upload a file**

```bash
curl -X POST http://localhost:5000/api/ingest/file \
  -F "file=@docs/paper.pdf"
```

**Check status**

```bash
curl http://localhost:5000/api/status
# {"ok": true, "chunks_indexed": 224, "ollama_model": "mistral", "embedding_model": "..."}
```

---

## CLI Usage

All CLI scripts remain fully supported alongside the web UI.

```bash
# Ingest a PDF
python scripts/ingest.py --source docs/paper.pdf

# Ingest a web page
python scripts/ingest.py --source https://arxiv.org/abs/1706.03762

# Ingest multiple sources at once
python scripts/ingest.py --source paper1.pdf --source paper2.pdf

# Ingest all files in a folder
python scripts/ingest.py --folder docs/

# Reset the index and re-ingest from scratch
python scripts/ingest.py --source docs/paper.pdf --reset
```

Expected output:

```
Ingestion complete!
  Documents loaded : 21 page(s)
  Chunks created   : 224
  Total in store   : 224
```

```bash
# Ask a single question
python scripts/ask.py --question "What is retrieval augmented generation?"

# Interactive mode (recommended for exploration)
python scripts/ask.py --interactive

# JSON output for programmatic use
python scripts/ask.py --question "What are RAG challenges?" --json

# Verbose mode — shows retrieval pipeline stages
python scripts/ask.py --interactive --verbose
```

Example answer:

```
============================================================
QUESTION: What is retrieval augmented generation?
============================================================

ANSWER:
Retrieval Augmented Generation (RAG) is a framework that enhances
large language models by retrieving relevant information from external
knowledge sources before generating responses [Chunk 2]. It combines
the parametric knowledge of LLMs with non-parametric retrieval systems
to reduce hallucination and improve factual accuracy [Chunk 3].

------------------------------------------------------------
CITATIONS (2):
  [1] Chunk 2 | docs/RAG.pdf | Page 1
       Score: 0.934
       "RAG integrates retrieval mechanisms with generative models..."

  [2] Chunk 3 | docs/RAG.pdf | Page 2
       Score: 0.871
       "The framework addresses the knowledge limitations of LLMs..."
============================================================
```

When information is not present in the indexed documents:

```
I cannot answer this question from the provided documents.
```

This is the hallucination gate working as intended — the system refuses to guess.

---

## Configuration

All parameters live in `config.py`. Change once, affects everywhere — no magic numbers scattered across the codebase.

```python
# Chunking
CHUNK_SIZE        = 600     # target tokens per chunk (sweet spot: 500–800)
CHUNK_OVERLAP     = 100     # overlap between adjacent chunks

# Retrieval
VECTOR_TOP_K      = 20      # candidates from vector search
BM25_TOP_K        = 20      # candidates from BM25 keyword search
RRF_K_CONSTANT    = 60      # RRF fusion constant (from TREC research)
RERANK_TOP_K      = 5       # chunks sent to the LLM after reranking

# Models
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DEVICE  = "cpu"   # change to "cuda" for GPU acceleration
OLLAMA_MODEL      = "mistral"

# LLM behaviour
LLM_TEMPERATURE   = 0.1     # lower = more factual, less creative
LLM_MAX_TOKENS    = 1024

# CI/CD quality gates
MIN_FAITHFULNESS        = 0.85
MIN_ANSWER_RELEVANCY    = 0.80
MIN_CONTEXT_PRECISION   = 0.75
```

### Swapping the LLM

Change one line in `config.py`:

```python
OLLAMA_MODEL = "mistral"          # default — 7B, best instruction following
OLLAMA_MODEL = "phi3:mini"        # fastest on CPU — 3.8B parameters
OLLAMA_MODEL = "gemma2:9b"        # best quality — requires ~6GB RAM
OLLAMA_MODEL = "deepseek-r1:7b"   # strong reasoning capability
```

### Prompt Versioning

Prompts live in `prompts/v1.yaml` and are git-tracked. To update a prompt safely:

1. Copy `prompts/v1.yaml` to `prompts/v2.yaml`
2. Edit the new file
3. Set `PROMPT_VERSION = "v2"` in `config.py`
4. Run evaluation: `python scripts/evaluate.py`
5. If faithfulness improves — commit. If it regresses — revert.

This workflow ensures every prompt change is logged, reviewable, and reversible.

---

## Evaluation

### Create Your Golden Dataset

Add verified question-answer pairs to `eval/golden_dataset.json`:

```json
[
  {
    "question": "What are the three paradigms of RAG?",
    "ground_truth": "The three paradigms are Naive RAG, Advanced RAG, and Modular RAG."
  },
  {
    "question": "What is the purpose of a reranker?",
    "ground_truth": "A reranker rescores retrieved candidates for precision after the initial broad retrieval."
  }
]
```

Aim for 50–200 pairs. The more verified pairs you have, the more reliable the benchmark and the more meaningful a CI failure becomes.

### Run Evaluation Locally

```bash
python scripts/evaluate.py --golden eval/golden_dataset.json
```

Example report:

```
================================================================
RAGAS EVALUATION REPORT
================================================================
Metric                    Score     Threshold    Status
----------------------------------------------------------------
  faithfulness            0.9120         0.85     PASS
  answer_relevancy        0.8830         0.80     PASS
  context_precision       0.7940         0.75     PASS
----------------------------------------------------------------

Overall gate:  PASSED
================================================================
```

### CI/CD — Automatic Quality Gate

Every pull request triggers a full RAGAS evaluation via GitHub Actions (`.github/workflows/rag_quality_gate.yml`). If any metric drops below its threshold, the PR is blocked from merging and the workflow posts the full results as a comment on the PR.

This is how production AI teams prevent silent quality regressions — the same discipline applied to model quality that developers already apply to unit tests.

---

## Running Tests

```bash
# Run all 15 unit tests
python -m pytest tests/ -v

# Run a specific test class
python -m pytest tests/test_pipeline.py::TestCitationEnforcement -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing
```

| Test Class | Tests | Coverage |
|---|---|---|
| `TestChunking` | 4 | Chunk size bounds, metadata injection, unique IDs, overlap |
| `TestBM25` | 3 | Search results, exact match ranking, save/load roundtrip |
| `TestRRFFusion` | 2 | Score decreases with rank, double-rank beats single |
| `TestCitationEnforcement` | 4 | Valid citation, no-citation block, cannot-answer gate, out-of-range ref |
| `TestContextBuilder` | 1 | `[Chunk N]` headers present in assembled context |
| | **15 total** | **All passing** |

---

## Key Design Decisions

**Why BM25 and vector search instead of vector search alone?**
Vector search understands meaning but misses exact terms — model names like `GPT-4o`, version strings like `BERT-large-uncased`, or rare acronyms like `RLHF`. BM25 handles these precisely. Combining both with RRF covers what neither can alone, typically improving retrieval recall by 15–25%.

**Why a cross-encoder reranker as a second stage?**
Bi-encoders encode query and document separately — they cannot model the interaction between them. Cross-encoders read both together through self-attention, capturing nuances like negation, contrast, and specificity. They are too slow for a full corpus but ideal for rescoring 20 candidates down to 5.

**Why store prompts in YAML files?**
Prompts are architecture. A prompt change that degrades faithfulness from 0.91 to 0.79 is a regression as serious as a code bug. Storing prompts in git means every change is logged, reviewable, and reversible — the difference between a research experiment and a production system.

**Why Mistral 7B over smaller models?**
Mistral's Grouped Query Attention (GQA) and Sliding Window Attention (SWA) give it exceptional instruction-following quality at the 7B scale. For a citation-heavy RAG system, reliably producing `[Chunk N]` exactly where instructed matters more than raw throughput.

**Why ChromaDB over Pinecone or Weaviate?**
Zero infrastructure. ChromaDB persists to local SQLite — restart the process and the data is still there. No account, no API key, no billing, no data leaving the machine. For a local-first system, this is the right default. Swapping to Weaviate or Pinecone later requires changing only `src/vectorstore.py`.

---

## Swapping Components

The pipeline is designed to be modular. Each stage is independent and can be replaced without touching the others.

- **Different LLM** — change `OLLAMA_MODEL` in `config.py` to any model Ollama supports
- **Different embedding model** — change `EMBEDDING_MODEL` in `config.py`
- **Different vector store** — swap `src/vectorstore.py` for a Weaviate, Qdrant, or Pinecone client with the same interface
- **Cloud LLM** — replace `src/generator.py` with an OpenAI or Anthropic API client
- **Add query expansion** — insert a query rewriting step before Stage 1 in `src/pipeline.py`
- **Add re-retrieval loop** — add a confidence check after Stage 5 and re-retrieve if the score is low

---

## Troubleshooting

**"No documents indexed yet"**
At least one document must be ingested before asking questions. Run `python scripts/ingest.py --source <file_or_url>`, or use the web UI to upload a file directly.

**"Cannot copy out of meta tensor"**
This is a PyTorch / Transformers version mismatch. `app.py` applies the fix automatically at startup via `torch.set_default_dtype(torch.float32)`. If you encounter it in a standalone script, add these two lines before any other import:

```python
import torch
torch.set_default_dtype(torch.float32)
```

**Ollama connection refused**
The Ollama server must be running before `app.py` starts. Run `ollama serve` in a separate terminal, then start the application.

**Slow responses on CPU**
Expected. Mistral 7B on CPU generates approximately 2–5 tokens per second. For faster responses, enable GPU acceleration (`EMBEDDING_DEVICE = "cuda"` in `config.py`) or switch to a smaller model (`ollama pull phi3:mini` and set `OLLAMA_MODEL = "phi3:mini"`).

**BM25 index out of sync after manual ChromaDB edits**
If you modify ChromaDB directly, rebuild the BM25 index by deleting `chroma_db/bm25_index.pkl` and restarting the application — it will rebuild automatically on the next startup.

---

## Roadmap

- [ ] Query expansion — generate alternative phrasings before retrieval to improve recall
- [ ] Multi-document comparison — answer questions that span multiple source documents
- [ ] Table and figure extraction — parse structured data from PDFs using PyMuPDF and camelot
- [ ] Re-retrieval loop — if answer confidence is low, retrieve again with a refined query
- [ ] Multi-modal ingestion — ingest images and diagrams alongside text via vision models in Ollama
- [ ] Weaviate backend — optional swap for production-scale deployments

---

## References

This system implements ideas from the following research:

- **Attention Is All You Need** — Vaswani et al. (2017) — Transformer architecture
- **Dense Passage Retrieval** — Karpukhin et al. (2020) — bi-encoder retrieval
- **Retrieval-Augmented Generation** — Lewis et al. (2020) — original RAG paper
- **Reciprocal Rank Fusion** — Cormack et al. (2009) — hybrid retrieval fusion
- **MS MARCO** — Nguyen et al. (2016) — reranker training dataset
- **RAGAS** — Es et al. (2023) — RAG evaluation framework
- **Self-RAG** — Asai et al. (2023) — adaptive retrieval with reflection tokens
- **Survey on RAG** — Gao et al. (2024) — comprehensive RAG taxonomy

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

*If it is not grounded, it is a guess.*

</div>
