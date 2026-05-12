"""
scripts/ingest.py — CLI for ingesting documents.

Usage:
  python scripts/ingest.py --source docs/paper.pdf
  python scripts/ingest.py --source https://arxiv.org/abs/1706.03762
  python scripts/ingest.py --folder docs/
  python scripts/ingest.py --source paper1.pdf --source paper2.pdf --reset
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.ingestion  import ingest, load_folder, chunk_documents, load_source
from src.vectorstore import add_chunks, get_chunk_count
from src.bm25_index  import get_bm25_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into Ask My Docs vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single PDF
  python scripts/ingest.py --source docs/attention.pdf

  # Ingest an arXiv page
  python scripts/ingest.py --source https://arxiv.org/abs/1706.03762

  # Ingest all docs in a folder
  python scripts/ingest.py --folder docs/

  # Reset everything and re-ingest
  python scripts/ingest.py --folder docs/ --reset
        """
    )
    parser.add_argument(
        "--source", action="append", dest="sources", metavar="PATH_OR_URL",
        help="PDF, Markdown file, or URL. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--folder", type=str, default=None,
        help="Ingest all supported files from this folder.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe existing vector store before ingesting (start fresh).",
    )
    args = parser.parse_args()

    if not args.sources and not args.folder:
        parser.print_help()
        sys.exit(1)

    # ── Load documents ────────────────────────────────────────────────────────
    all_docs = []

    if args.folder:
        logger.info(f"Loading all documents from folder: {args.folder}")
        from src.ingestion import load_folder, chunk_documents
        folder_docs = load_folder(args.folder)
        all_docs.extend(folder_docs)

    if args.sources:
        for source in args.sources:
            logger.info(f"Loading: {source}")
            docs = load_source(source)
            all_docs.extend(docs)

    if not all_docs:
        logger.error("No documents loaded. Check your paths/URLs.")
        sys.exit(1)

    logger.info(f"Loaded {len(all_docs)} raw document page(s).")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    from src.ingestion import chunk_documents
    chunks = chunk_documents(all_docs)
    logger.info(f"Created {len(chunks)} chunks "
                f"({config.CHUNK_SIZE} tokens, {config.CHUNK_OVERLAP} overlap).")

    # ── Embed + store in ChromaDB ─────────────────────────────────────────────
    logger.info("Embedding chunks and storing in ChromaDB (this may take a minute)...")
    stored = add_chunks(chunks, reset=args.reset)
    total = get_chunk_count()
    logger.info(f"Stored {stored} new chunks. Total in store: {total}.")

    # ── Rebuild BM25 index ────────────────────────────────────────────────────
    logger.info("Rebuilding BM25 keyword index...")
    from src.vectorstore import get_all_chunks
    all_indexed = get_all_chunks()
    get_bm25_index(chunks=all_indexed, force_rebuild=True)
    logger.info("BM25 index saved.")

    print(f"\n✅  Ingestion complete!")
    print(f"   Documents loaded : {len(all_docs)} page(s)")
    print(f"   Chunks created   : {len(chunks)}")
    print(f"   Total in store   : {total}")
    print(f"\nNow ask questions:")
    print(f"   python scripts/ask.py --question \"What is attention?\"")
    print(f"   python scripts/ask.py --interactive")


if __name__ == "__main__":
    main()
