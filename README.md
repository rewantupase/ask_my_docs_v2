# Ask My Docs — Free RAG System

A production-grade, 100% free "Ask My Docs" system.
Supports PDF, Markdown, and web pages. Hybrid BM25 + vector retrieval,
cross-encoder reranking, citation enforcement, and RAGAS evaluation.

---

## Stack (all free, all local)

| Layer            | Tool                              |
|------------------|-----------------------------------|
| Embeddings       | sentence-transformers all-MiniLM  |
| Vector store     | ChromaDB (local SQLite)           |
| Keyword search   | rank-bm25                         |
| Reranker         | cross-encoder/ms-marco-MiniLM     |
| LLM              | Ollama + Mistral 7B v0.3 (local)  |
| Evaluation       | RAGAS (offline)                   |
| Orchestration    | LangChain                         |

---

## Setup (one time)

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama and pull Llama 3.2
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral            # Mistral 7B v0.3, ~4GB, runs on CPU
```

### 3. Add your documents
Drop any PDF, Markdown, or paste a URL into docs/ or pass directly to the CLI.

---

## Usage

### Ingest documents
```bash
# Ingest a PDF
python scripts/ingest.py --source docs/attention_is_all_you_need.pdf

# Ingest a web page
python scripts/ingest.py --source https://arxiv.org/abs/1706.03762

# Ingest all docs in a folder
python scripts/ingest.py --folder docs/
```

### Ask questions
```bash
python scripts/ask.py --question "What optimizer does the Transformer use?"

# Interactive mode
python scripts/ask.py --interactive
```

### Run evaluation
```bash
python scripts/evaluate.py --golden eval/golden_dataset.json
```

---

## Project structure
```
ask_my_docs/
├── README.md
├── requirements.txt
├── config.py                  # central config (model names, paths, thresholds)
├── prompts/
│   └── v1.yaml                # versioned prompt templates
├── src/
│   ├── __init__.py
│   ├── ingestion.py           # load + chunk documents
│   ├── embeddings.py          # embedding model wrapper
│   ├── vectorstore.py         # ChromaDB operations
│   ├── bm25_index.py          # BM25 keyword index
│   ├── retriever.py           # hybrid retrieval + RRF fusion
│   ├── reranker.py            # cross-encoder reranker
│   ├── generator.py           # LLM answer generation
│   ├── citations.py           # citation enforcement + validation
│   └── pipeline.py            # full RAG pipeline (wires everything)
├── scripts/
│   ├── ingest.py              # CLI: ingest documents
│   ├── ask.py                 # CLI: ask questions
│   └── evaluate.py            # CLI: run RAGAS evaluation
├── eval/
│   └── golden_dataset.json    # your verified QA pairs
├── tests/
│   └── test_pipeline.py       # unit tests
└── docs/                      # drop your documents here
```
