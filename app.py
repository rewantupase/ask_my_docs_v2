"""
app.py — Flask web UI for Ask My Docs.
"""

# ── Torch meta-tensor fix ─────────────────────────────────────────────────────
# MUST be before any other imports. Prevents "Cannot copy out of meta tensor"
# when sentence-transformers loads models on newer torch/transformers versions.
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_default_dtype(torch.float32)
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import threading
from pathlib import Path
import sys

from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context

sys.path.insert(0, str(Path(__file__).parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="web/static", template_folder="web")

# ── Pipeline singleton ────────────────────────────────────────────────────────

_pipeline_lock = threading.Lock()
_global_pipeline = None


def get_pipeline():
    global _global_pipeline
    with _pipeline_lock:
        if _global_pipeline is None:
            from src.pipeline import build_pipeline
            _global_pipeline = build_pipeline()
    return _global_pipeline


def invalidate_pipeline():
    global _global_pipeline
    with _pipeline_lock:
        _global_pipeline = None


# ── Document registry ─────────────────────────────────────────────────────────

def get_indexed_sources() -> list[dict]:
    try:
        from src.vectorstore import get_all_chunks
        chunks = get_all_chunks()
    except Exception:
        return []

    source_counts: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    docs = []
    for src, count in sorted(source_counts.items()):
        name = Path(src).name if not src.startswith("http") else src
        docs.append({"name": name, "source": src, "chunk_count": count})
    return docs


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/api/docs", methods=["GET"])
def list_docs():
    docs = get_indexed_sources()
    return jsonify({"docs": docs, "total": len(docs)})


@app.route("/api/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()

    if not question:
        return jsonify({"error": "question is required"}), 400

    def generate():
        try:
            pipeline = get_pipeline()
        except RuntimeError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        try:
            response = pipeline(question)
        except Exception as e:
            logger.exception("Pipeline error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        result = {
            "answer": response.answer,
            "cannot_answer": response.cannot_answer,
            "was_blocked": response.was_blocked,
            "citations": [
                {
                    "chunk_number": c.chunk_number,
                    "source": c.source,
                    "source_name": Path(c.source).name if not c.source.startswith("http") else c.source,
                    "page": c.page,
                    "preview": c.preview[:200],
                    "score": round(c.relevance_score, 4),
                }
                for c in response.citations
            ],
        }
        yield f"data: {json.dumps(result)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/ingest/file", methods=["POST"])
def ingest_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    filename = f.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in (".pdf", ".md", ".markdown"):
        return jsonify({"error": f"Unsupported file type: {suffix}. Use PDF or Markdown."}), 400

    save_dir = config.DOCS_DIR
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / filename
    f.save(str(save_path))
    logger.info(f"Saved upload: {save_path}")

    try:
        from src.ingestion import ingest
        from src.vectorstore import add_chunks, get_chunk_count, get_all_chunks
        from src.bm25_index import get_bm25_index

        chunks = ingest(str(save_path))
        stored = add_chunks(chunks)
        all_chunks = get_all_chunks()
        get_bm25_index(chunks=all_chunks, force_rebuild=True)
        invalidate_pipeline()
        total = get_chunk_count()

        return jsonify({
            "ok": True,
            "filename": filename,
            "chunks_created": len(chunks),
            "chunks_stored": stored,
            "total_chunks": total,
        })
    except Exception as e:
        logger.exception("Ingestion error")
        save_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest/url", methods=["POST"])
def ingest_url():
    body = request.get_json(force=True)
    url = (body.get("url") or "").strip()

    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "url must start with http:// or https://"}), 400

    try:
        from src.ingestion import ingest
        from src.vectorstore import add_chunks, get_chunk_count, get_all_chunks
        from src.bm25_index import get_bm25_index

        chunks = ingest(url)
        stored = add_chunks(chunks)
        all_chunks = get_all_chunks()
        get_bm25_index(chunks=all_chunks, force_rebuild=True)
        invalidate_pipeline()
        total = get_chunk_count()

        return jsonify({
            "ok": True,
            "url": url,
            "chunks_created": len(chunks),
            "chunks_stored": stored,
            "total_chunks": total,
        })
    except Exception as e:
        logger.exception("URL ingestion error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/docs/<path:source_name>", methods=["DELETE"])
def delete_doc(source_name):
    try:
        from src.vectorstore import get_vectorstore, get_all_chunks
        from src.bm25_index import get_bm25_index

        store = get_vectorstore()
        collection = store._collection

        all_data = collection.get(include=["metadatas"])
        ids_to_delete = [
            id_ for id_, meta in zip(all_data["ids"], all_data["metadatas"])
            if Path(meta.get("source", "")).name == source_name
            or meta.get("source", "") == source_name
        ]

        if not ids_to_delete:
            return jsonify({"error": f"No chunks found for '{source_name}'"}), 404

        collection.delete(ids=ids_to_delete)
        all_chunks = get_all_chunks()
        get_bm25_index(chunks=all_chunks, force_rebuild=True)
        invalidate_pipeline()

        return jsonify({
            "ok": True,
            "deleted_chunks": len(ids_to_delete),
            "source": source_name,
        })
    except Exception as e:
        logger.exception("Delete error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    try:
        from src.vectorstore import get_chunk_count
        count = get_chunk_count()
    except Exception:
        count = 0

    return jsonify({
        "ok": True,
        "chunks_indexed": count,
        "ollama_model": config.OLLAMA_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
    })


if __name__ == "__main__":
    print("\n🔍  Ask My Docs — Web UI")
    print("   Open http://localhost:5000 in your browser\n")

    # ── Pre-warm singletons before Flask starts serving ───────────────────────
    # Prevents race conditions: embedding model and ChromaDB are fully
    # initialised in the main thread before any request thread can touch them.
    logger.info("Pre-loading embedding model and vector store...")
    try:
        from src.embeddings import get_embedding_model
        from src.vectorstore import get_vectorstore
        get_embedding_model()
        get_vectorstore()
        logger.info("Pre-load complete.")
    except Exception as e:
        logger.warning(f"Pre-load skipped: {e}")
    # ─────────────────────────────────────────────────────────────────────────

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)