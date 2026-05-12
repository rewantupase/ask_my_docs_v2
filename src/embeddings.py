"""
src/embeddings.py

Root cause of meta-tensor error: SentenceTransformer.__init__ calls
self.to(device) after transformers loads weights onto meta device for
dtype sniffing. torch.nn.Module.to() raises NotImplementedError on meta tensors.

Fix: monkey-patch torch.nn.Module.to() inside _force_cpu_alloc() so that
when a module has meta tensors it uses to_empty() instead of to().
A threading lock prevents double-loading under Flask's threaded mode.
"""

import logging
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import List
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()   # prevents double-load under threaded Flask


@contextmanager
def _force_cpu_alloc():
    """
    Patches torch allocation functions AND nn.Module.to() so that any
    meta-device allocation or move is redirected to CPU instead of crashing.
    Restores all originals on exit.
    """
    # --- patch 1: torch allocation functions ---
    _fns = ["empty", "zeros", "ones", "full"]
    originals = {name: getattr(torch, name) for name in _fns}

    def _make_patched(orig):
        def _patched(*args, **kwargs):
            if str(kwargs.get("device")) == "meta":
                kwargs["device"] = "cpu"
            return orig(*args, **kwargs)
        return _patched

    for name, orig in originals.items():
        setattr(torch, name, _make_patched(orig))

    # --- patch 2: nn.Module.to() ---
    # This is the real fix: SentenceTransformer calls self.to(device) after
    # from_pretrained, which crashes when weights are on meta device.
    # to_empty() allocates fresh (uninitialized) CPU memory without copying.
    original_to = nn.Module.to

    def _safe_to(self, *args, **kwargs):
        has_meta = any(
            p.is_meta
            for p in list(self.parameters()) + list(self.buffers())
        )
        if not has_meta:
            return original_to(self, *args, **kwargs)

        # Determine target device from args/kwargs
        device = None
        if args:
            first = args[0]
            if isinstance(first, (str, torch.device)):
                device = torch.device(first)
        if "device" in kwargs:
            device = torch.device(kwargs["device"])

        if device is None:
            # dtype-only call or unknown — fall through
            return original_to(self, *args, **kwargs)

        # Safe path: allocate empty tensors on target device
        self.to_empty(device=device)
        return self

    nn.Module.to = _safe_to

    try:
        yield
    finally:
        for name, orig in originals.items():
            setattr(torch, name, orig)
        nn.Module.to = original_to


@lru_cache(maxsize=1)
def _load_st_model():
    from sentence_transformers import SentenceTransformer

    with _model_lock:   # only one thread loads; lru_cache handles subsequent calls
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")

        with _force_cpu_alloc():
            st = SentenceTransformer(
                config.EMBEDDING_MODEL,
                device="cpu",
            )

        # Belt-and-suspenders: materialize any remaining meta params
        for name, param in list(st.named_parameters()):
            if param.is_meta:
                logger.warning(f"Materializing meta param: {name}")
                param.data = torch.empty(param.shape, dtype=param.dtype, device="cpu")

        # Belt-and-suspenders: materialize any remaining meta buffers
        for name, buf in list(st.named_buffers()):
            if buf.is_meta:
                logger.warning(f"Materializing meta buffer: {name}")
                parts = name.rsplit(".", 1)
                parent = dict(st.named_modules())[parts[0]] if len(parts) > 1 else st
                setattr(parent, parts[-1],
                        torch.empty(buf.shape, dtype=buf.dtype, device="cpu"))

        # Weights are now real float32 on CPU — safe to move to target device
        if config.EMBEDDING_DEVICE != "cpu":
            st = st.to(config.EMBEDDING_DEVICE)

        logger.info("Embedding model loaded.")
        return st


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    LangChain-compatible Embeddings wrapper backed by our safely-loaded
    SentenceTransformer. Custom subclass avoids HuggingFaceEmbeddings
    re-constructing SentenceTransformer internally and bypassing our fix.
    """
    from langchain_core.embeddings import Embeddings

    st = _load_st_model()

    class _SafeEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            vecs = st.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vecs.tolist()

        def embed_query(self, text: str) -> List[float]:
            vec = st.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vec.tolist()

    return _SafeEmbeddings()


def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embedding_model().embed_documents(texts)


def embed_query(query: str) -> List[float]:
    return get_embedding_model().embed_query(query)