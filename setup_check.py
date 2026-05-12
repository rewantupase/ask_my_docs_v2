#!/usr/bin/env python3
"""
setup_check.py — Verify your environment is ready to run Ask My Docs.

Run this after: pip install -r requirements.txt

It checks:
  1. All required Python packages are importable
  2. Ollama is installed and running
  3. llama3.2 model is available
  4. The embedding model can be loaded (downloads if not cached)
  5. ChromaDB can be created
  6. BM25 can be built
  7. Unit tests pass

Usage:
  python setup_check.py
"""

import sys
import subprocess

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "


def check(name: str, fn) -> bool:
    print(f"\n[{name}]", end="", flush=True)
    try:
        result = fn()
        msg = f" — {result}" if result else ""
        print(f"{PASS}{msg}")
        return True
    except Exception as e:
        print(f"{FAIL} {e}")
        return False


# ── Checks ────────────────────────────────────────────────────────────────────

def check_python():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"Python 3.10+ required, got {v.major}.{v.minor}")
    return f"Python {v.major}.{v.minor}.{v.micro}"


def check_langchain():
    import langchain
    return f"langchain {langchain.__version__}"


def check_chromadb():
    import chromadb
    return f"chromadb {chromadb.__version__}"


def check_sentence_transformers():
    import sentence_transformers
    return f"sentence-transformers {sentence_transformers.__version__}"


def check_rank_bm25():
    import rank_bm25
    return "rank-bm25 OK"


def check_ragas():
    import ragas
    return f"ragas {ragas.__version__}"


def check_pymupdf():
    import fitz   # PyMuPDF's import name
    return f"PyMuPDF {fitz.version[0]}"


def check_yaml():
    import yaml
    return "pyyaml OK"


def check_ollama_installed():
    result = subprocess.run(
        ["ollama", "--version"],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode != 0:
        raise RuntimeError("ollama not found. Install from https://ollama.com")
    return result.stdout.strip()


def check_ollama_running():
    import requests
    try:
        r = requests.get("http://localhost:11434/", timeout=3)
        if r.status_code == 200:
            return "Ollama server is running"
    except Exception:
        pass
    raise RuntimeError(
        "Ollama server not running. Start it with: ollama serve\n"
        "  (Or it auto-starts on macOS when you run ollama run ...)"
    )


def check_llama_model():
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True, text=True, timeout=10
    )
    if "mistral" not in result.stdout:
        raise RuntimeError(
            "mistral model not found.\n"
            "  Pull it with: ollama pull mistral"
        )
    return "mistral available"


def check_embedding_model():
    print("\n[Embedding model] Loading all-MiniLM-L6-v2...", end="", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vec = model.encode(["test sentence"])
    assert vec.shape == (1, 384), f"Expected (1, 384), got {vec.shape}"
    return "all-MiniLM-L6-v2 OK, 384-dim vectors"


def check_chromadb_create():
    import chromadb
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        client = chromadb.PersistentClient(path=tmpdir)
        col = client.get_or_create_collection("test")
        col.add(
            documents=["hello world"],
            ids=["test_id"],
            embeddings=[[0.1] * 384],
        )
        results = col.query(query_embeddings=[[0.1] * 384], n_results=1)
        assert results["ids"][0][0] == "test_id"
    return "ChromaDB read/write OK"


def check_bm25():
    from rank_bm25 import BM25Okapi
    corpus = [["hello", "world"], ["foo", "bar"], ["bm25", "works"]]
    index = BM25Okapi(corpus)
    scores = index.get_scores(["bm25"])
    import numpy as np
    best = np.argmax(scores)
    assert best == 2, f"BM25 returned wrong best match: {best}"
    return "BM25 search OK"


def check_unit_tests():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_pipeline.py", "-v", "--tb=short", "-q"],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Some tests failed:\n{result.stdout[-800:]}\n{result.stderr[-400:]}"
        )
    lines = result.stdout.strip().split("\n")
    summary = lines[-1] if lines else "tests ran"
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Ask My Docs — Environment Check")
    print("=" * 60)

    results = []

    # Python packages (must all pass)
    print("\n── Python packages ──")
    results += [
        check("Python version",          check_python),
        check("langchain",               check_langchain),
        check("chromadb",                check_chromadb),
        check("sentence-transformers",   check_sentence_transformers),
        check("rank-bm25",               check_rank_bm25),
        check("ragas",                   check_ragas),
        check("pymupdf",                 check_pymupdf),
        check("pyyaml",                  check_yaml),
    ]

    # Ollama (warn if not running — required for ask.py, not for tests)
    print("\n── Ollama (local LLM) ──")
    ollama_ok = check("ollama installed", check_ollama_installed)
    if ollama_ok:
        running = check("ollama server",  check_ollama_running)
        if running:
            check("mistral model",        check_llama_model)
        else:
            print(f"{WARN} Run 'ollama serve' before using ask.py")
    else:
        print(f"{WARN} Install Ollama from https://ollama.com")

    # Functional checks
    print("\n── Functional checks ──")
    results += [
        check("Embedding model",          check_embedding_model),
        check("ChromaDB read/write",      check_chromadb_create),
        check("BM25 search",              check_bm25),
        check("Unit tests",               check_unit_tests),
    ]

    # Summary
    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉  All checks passed! You're ready to run Ask My Docs.")
        print("\nNext steps:")
        print("  1. Drop a PDF into docs/  (or use a URL)")
        print("  2. python scripts/ingest.py --source docs/your_paper.pdf")
        print("  3. python scripts/ask.py --interactive")
    else:
        print(f"\n⚠️   {total - passed} check(s) failed. Fix the errors above, then re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
