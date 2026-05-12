"""
src/vectorstore.py — ChromaDB operations.

ChromaDB is a local, SQLite-backed vector database.
All data lives in ./chroma_db/ — no cloud, no API key, no network calls.
"""

import logging
import threading
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.embeddings import get_embedding_model

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_vectorstore = None
_vectorstore_lock = threading.Lock()   # guards ChromaDB init against race conditions


def get_vectorstore(reset: bool = False):
    """
    Get (or create) the ChromaDB vector store.
    Singleton: only opens the DB connection once per process.
    Lock prevents two threads both seeing _vectorstore=None and both
    trying to create a PersistentClient simultaneously (causes KeyError).
    """
    global _vectorstore
    with _vectorstore_lock:
        if _vectorstore is not None and not reset:
            return _vectorstore

        from langchain_chroma import Chroma

        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        embedding_model = get_embedding_model()

        if reset:
            import shutil
            if config.CHROMA_DIR.exists():
                shutil.rmtree(config.CHROMA_DIR)
            config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("ChromaDB collection reset.")

        _vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=str(config.CHROMA_DIR),
        )
        count = _vectorstore._collection.count()
        logger.info(f"ChromaDB ready — {count} chunks indexed.")
        return _vectorstore


def add_chunks(chunks: List[Document], reset: bool = False) -> int:
    """
    Embed and store chunk Documents in ChromaDB.
    Uses upsert (via add_documents) so re-ingesting the same doc is safe —
    existing chunk IDs are updated, not duplicated.
    Returns the number of chunks stored.
    """
    if not chunks:
        logger.warning("No chunks to add.")
        return 0

    store = get_vectorstore(reset=reset)
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    # Deduplicate within this batch — same chunk_id must not appear twice
    # in a single upsert call even if ChromaDB would normally handle it.
    seen = set()
    deduped_chunks = []
    deduped_ids = []
    for chunk, cid in zip(chunks, ids):
        if cid not in seen:
            seen.add(cid)
            deduped_chunks.append(chunk)
            deduped_ids.append(cid)

    if len(deduped_chunks) < len(chunks):
        logger.warning(
            f"Removed {len(chunks) - len(deduped_chunks)} duplicate chunk IDs "
            f"before upserting."
        )

    logger.info(f"Embedding and storing {len(deduped_chunks)} chunks...")

    # Use upsert directly so re-ingesting the same document overwrites
    # existing vectors rather than raising DuplicateIDError.
    collection = store._collection
    texts = [c.page_content for c in deduped_chunks]
    metadatas = [c.metadata for c in deduped_chunks]

    # Embed via the same model used by the store
    embeddings = store._embedding_function.embed_documents(texts)

    collection.upsert(
        ids=deduped_ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    logger.info(f"Stored {len(deduped_chunks)} chunks in ChromaDB.")
    return len(deduped_chunks)


def similarity_search(query: str, k: int = None) -> List[Tuple[Document, float]]:
    """Vector similarity search. Returns (Document, score) tuples."""
    k = k or config.VECTOR_TOP_K
    store = get_vectorstore()
    return store.similarity_search_with_relevance_scores(query, k=k)


def get_all_chunks() -> List[Document]:
    """Retrieve all stored chunks (used to rebuild BM25 index after restart)."""
    store = get_vectorstore()
    collection = store._collection
    result = collection.get(include=["documents", "metadatas"])

    docs = []
    for text, metadata in zip(result["documents"], result["metadatas"]):
        doc = Document(page_content=text, metadata=metadata or {})
        docs.append(doc)

    logger.info(f"Retrieved {len(docs)} chunks from ChromaDB.")
    return docs


def get_chunk_count() -> int:
    """Return the total number of chunks indexed."""
    store = get_vectorstore()
    return store._collection.count()